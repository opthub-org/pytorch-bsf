// Exercise the real engine bundled in Action v5.0.0, without GitHub.
const assert = require('node:assert/strict');
const {readFileSync} = require('node:fs');
const {resolve} = require('node:path');
const test = require('node:test');
const Ajv = require('ajv');
const YAML = require('yaml');
const {Manifest} = require('release-please');
const {buildStrategy} = require('release-please/build/src/factory');
const {parsePyProject} = require('release-please/build/src/updaters/python/pyproject-toml');

const root = resolve(__dirname, '../../..');
const read = path => readFileSync(resolve(root, path), 'utf8');
const config = JSON.parse(read('release-please-config.json'));
const versions = JSON.parse(read('.release-please-manifest.json'));
const workflow = YAML.parse(read('.github/workflows/release-please-action.yml'));
const baseline = '0.24.0';
// Existing, published v0.24.0: fixtures must stop at this commit, not restart history.
const releaseSha = '740397a6903e343e63ab3cf35bba26fb2d82b1b4';
const logger = Object.fromEntries(
  ['debug', 'info', 'warn', 'error', 'trace'].map(name => [name, () => {}]),
);
require('release-please/build/src/util/logger').setLogger(logger);

function offlineRepository(message, tagsOnly = false, version = baseline) {
  const files = {
    'release-please-config.json': JSON.stringify(config),
    '.release-please-manifest.json': JSON.stringify({'.': version}),
    'pyproject.toml': read('pyproject.toml').replace(
      /^version = ".*"$/m, `version = "${version}"`,
    ),
  };
  const scm = {
    repository: {owner: 'opthub-org', repo: 'pytorch-bsf', defaultBranch: 'master'},
    async getFileJson(path) { return JSON.parse(files[path]); },
    async getFileContentsOnBranch(path) {
      assert.ok(path in files, `Unexpected file read: ${path}`);
      return {parsedContent: files[path]};
    },
    async findFilesByFilenameAndRef() { return []; },
    async *releaseIterator() {
      if (!tagsOnly) yield {tagName: `v${version}`, sha: releaseSha};
    },
    async *tagIterator() { yield {name: `v${version}`, sha: releaseSha}; },
    async *mergeCommitIterator() {
      if (message) yield {sha: 'a'.repeat(40), message, files: ['torch_bsf/core.py']};
      yield {sha: releaseSha, message: `chore(master): release ${version}`, files: ['pyproject.toml']};
      assert.fail('Release Please scanned commits older than the published release');
    },
  };
  // No credentials, network client, or write methods are provided. Fail closed if
  // the fixture accidentally calls createReleases/createPullRequests or other I/O.
  return new Proxy(scm, {
    get(target, key) {
      assert.ok(key in target, `Unexpected SCM operation: ${String(key)}`);
      return target[key];
    },
  });
}

async function candidate(message, tagsOnly = false, version = baseline) {
  const scm = offlineRepository(message, tagsOnly, version);
  const manifest = await Manifest.fromManifest(scm, 'master', undefined, undefined, {logger});
  const prs = await manifest.buildPullRequests();
  assert.equal(prs.length, message ? 1 : 0);
  return {scm, manifest, pr: prs[0]};
}

test('configuration and current manifest satisfy the engine schemas and package metadata', () => {
  const ajv = new Ajv({strict: false, formats: {'uri-reference': true}});
  for (const [name, data] of [['config', config], ['manifest', versions]]) {
    const schema = require(`release-please/schemas/${name}.json`);
    const validate = ajv.compile(schema);
    assert.ok(validate(data), JSON.stringify(validate.errors));
  }
  assert.deepEqual(Object.keys(config.packages), ['.']);
  const project = parsePyProject(read('pyproject.toml')).project;
  assert.equal(project.name, 'pytorch-bsf');
  assert.equal(project.version, versions['.']);
  assert.ok(read('CHANGELOG.md').includes(`## [${versions['.']}]`));
  assert.equal(config.packages['.']['package-name'], 'torch_bsf');
  assert.equal(config.packages['.']['include-component-in-tag'], false);
  assert.equal(config.packages['.']['include-v-in-tag'], true);
  assert.equal(require('release-please/package.json').version, '17.6.0');
});

// Golden results verified against release-please 15.13.0 from Action v3.7.13.
// Keep these expectations without installing the legacy engine's vulnerable dependencies.
for (const [label, message, expected] of [
  ['breaking', 'feat!: remove a deprecated option\n\nBREAKING CHANGE: use its replacement.', '0.25.0'],
  ['feat', 'feat: add an option', '0.25.0'],
  ['fix', 'fix: preserve valid inputs', '0.24.1'],
]) {
  test(`${label}: preserve legacy versioning, Python updates, and unprefixed tags`, async () => {
    const {scm, manifest, pr} = await candidate(message);
    assert.equal(pr.version.toString(), expected);
    assert.equal(pr.headRefName, 'release-please--branches--master--components--torch_bsf');
    const strategy = await buildStrategy({
      github: scm, targetBranch: 'master', ...manifest.repositoryConfig['.'],
    });
    assert.equal(await strategy.getPackageName(), 'torch_bsf');
    assert.equal(await strategy.getComponent(), '');

    const updates = new Map(pr.updates.map(update => [update.path, update.updater]));
    const updatedProject = parsePyProject(updates.get('pyproject.toml').updateContent(read('pyproject.toml'))).project;
    assert.equal(updatedProject.version, expected);
    assert.equal(updatedProject.name, 'pytorch-bsf');
    const changelog = updates.get('CHANGELOG.md').updateContent(read('CHANGELOG.md'));
    assert.ok(changelog.includes(`/compare/v${baseline}...v${expected}`));
    assert.ok(changelog.includes(`## [${baseline}]`), 'Keep previous changelog entries');
    assert.ok(!changelog.includes('torch_bsf-v'), 'Keep existing v0.x tag format');
    const updatedVersions = JSON.parse(
      updates.get('.release-please-manifest.json').updateContent(JSON.stringify({'.': baseline})),
    );
    assert.equal(updatedVersions['.'], expected);
  });
}

test('existing tags recover the release boundary without a GitHub Release', async () => {
  const {pr} = await candidate('fix: preserve tag history', true);
  assert.equal(pr.version.toString(), '0.24.1');
});

test('no new commits produce no release PR or publication side effects', async () => {
  await candidate(undefined);
});

test('breaking changes after 1.0 keep the documented major bump', async () => {
  const {pr} = await candidate('feat!: change the API', false, '1.2.3');
  assert.equal(pr.version.toString(), '2.0.0');
});

test('Action inputs and root outputs connect only created releases to publishing', () => {
  const release = workflow.jobs['github-release'];
  const step = release.steps.find(step => step.id === 'release-please');
  assert.equal(step.uses, 'googleapis/release-please-action@45996ed1f6d02564a971a2fa1b5860e934307cf7');
  // v5 action.yml supports these inputs; supplying release-type would bypass the
  // manifest and discard its package-name and pre-major versioning policy.
  assert.deepEqual(step.with, {
    'config-file': 'release-please-config.json',
    'manifest-file': '.release-please-manifest.json',
  });
  assert.deepEqual(release.outputs, {
    release_created: '${{ steps.release-please.outputs.release_created }}',
    tag_name: '${{ steps.release-please.outputs.tag_name }}',
  });
  for (const name of ['docker', 'pypi-build']) {
    const job = workflow.jobs[name];
    assert.equal(job.needs, 'github-release');
    assert.equal(job.if, "${{ needs.github-release.outputs.release_created == 'true' }}");
  }
  assert.equal(workflow.jobs.docker.with.tag_name, '${{ needs.github-release.outputs.tag_name }}');
  const checkout = workflow.jobs['pypi-build'].steps.find(step => step.uses?.startsWith('actions/checkout@'));
  assert.equal(checkout.with.ref, '${{ needs.github-release.outputs.tag_name }}');
  const docker = YAML.parse(read('.github/workflows/docker.yml'));
  assert.equal(docker.on.workflow_call.inputs.tag_name.required, true);
  assert.equal(docker.jobs['build-and-push'].steps[0].with.ref,
    '${{ inputs.tag_name || github.event.release.tag_name }}');
  assert.deepEqual(workflow.on, {push: {branches: ['master']}});
});

test('PyPI publication requires a successful release build and only grants OIDC to the publisher', () => {
  const build = workflow.jobs['pypi-build'];
  const publish = workflow.jobs.pypi;
  assert.deepEqual(workflow.permissions, {});
  assert.deepEqual(build.permissions, {contents: 'read'});
  assert.deepEqual(publish.permissions, {'id-token': 'write'});
  for (const [name, job] of Object.entries(workflow.jobs)) {
    if (name !== 'pypi') assert.equal(job.permissions?.['id-token'], undefined);
  }
  assert.deepEqual(publish.needs, ['github-release', 'pypi-build']);
  assert.equal(publish.if, "${{ needs.github-release.outputs.release_created == 'true' }}");
  assert.deepEqual(publish.environment, {
    name: 'pypi', url: 'https://pypi.org/p/pytorch-bsf',
  });

  // The privileged job only transfers and publishes files; it must not run the
  // package's build backend, install dependencies, or execute repository scripts.
  assert.equal(publish.steps.length, 2);
  assert.ok(publish.steps.every(step => step.uses && !step.run));
  const [download, upload] = publish.steps;
  assert.match(download.uses, /^actions\/download-artifact@[0-9a-f]{40}$/);
  assert.match(upload.uses, /^pypa\/gh-action-pypi-publish@[0-9a-f]{40}$/);
  assert.deepEqual(upload.with, {attestations: true});
  assert.ok(!read('.github/workflows/release-please-action.yml').includes('PYPI_API_TOKEN'));
});

test('the publisher consumes only the immutable distribution artifact from its own build', () => {
  const build = workflow.jobs['pypi-build'];
  const artifact = build.steps.find(step => step.uses?.startsWith('actions/upload-artifact@'));
  assert.match(artifact.uses, /^actions\/upload-artifact@[0-9a-f]{40}$/);
  assert.equal(build.outputs['artifact-id'], '${{ steps.distributions.outputs.artifact-id }}');
  assert.equal(artifact.id, 'distributions');
  assert.equal(artifact.with.path, 'dist/*');
  assert.equal(artifact.with['if-no-files-found'], 'error');
  // A new attempt keeps the original artifact available when retrying a failed
  // publishing job; the output ID selects the artifact built for that attempt.
  assert.ok(artifact.with.name.includes('${{ github.run_attempt }}'));
  assert.deepEqual(workflow.jobs.pypi.steps[0].with, {
    'artifact-ids': '${{ needs.pypi-build.outputs.artifact-id }}',
    'merge-multiple': true,
    path: 'dist',
    'digest-mismatch': 'error',
  });
});
