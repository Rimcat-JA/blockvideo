import type { LanguageResponse } from '@/lib/language-types';

export function languageFixture(requestId = 'language-test', overrides: Partial<LanguageResponse> = {}): LanguageResponse {
  return {
    request_id: requestId, core_request_id: 'nl-core-test', mode: 'all_tools', status: 'completed', project_id: 4, base_revision: 3,
    interpretation: { status: 'proposed', executed: false, failure: null,
      proposal: { kind: 'operation', operation_id: 'project.subtitle-font-size.set', operation_version: 1, arguments: { value: 56 } } },
    clarification: null, prepared_request: { request_id: 'nl-core-test', operation_id: 'project.subtitle-font-size.set',
      operation_version: 1, target: { project_id: 4 }, arguments: { value: 56 }, base_revision: 3, generation_requested: false },
    requires_confirmation: false, confirmation_token: 'a'.repeat(64), failure: null, executed: true,
    result: { request_id: 'nl-core-test', operation_id: 'project.subtitle-font-size.set', project_id: 4, base_revision: 3,
      revision: 4, changed: true, resolved_arguments: { value: 56 }, generation_requested: false, job_id: null, data: { subtitle_font_size: 56 } },
    ...overrides,
  };
}

export function readyFixture(requestId = 'language-test'): LanguageResponse {
  return languageFixture(requestId, { status: 'ready', executed: false, result: null, requires_confirmation: true,
    prepared_request: { request_id: 'nl-core-test', operation_id: 'project.generation.start', operation_version: 1,
      arguments: { kind: 'full' }, target: { project_id: 4 }, base_revision: 3, generation_requested: false } });
}

export function compoundFixture(requestId = 'language-test'): LanguageResponse {
  const response = languageFixture(requestId);
  return { ...response, status: 'ready', generate_after_save: true, requires_confirmation: true,
    generation_request: { request_id: 'nl-core-test-generation', operation_id: 'project.generation.start',
      operation_version: 1, arguments: { kind: 'full' }, target: { project_id: 4 }, base_revision: 4,
      generation_requested: false }, generation_result: null };
}
