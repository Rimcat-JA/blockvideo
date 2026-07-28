/** React Query hooks that connect pages/components to the API client. */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import type { CreateProjectInput, QuickCreateInput } from '@/lib/types';

export function useProjects() {
  /** Fetch the newest project summaries. */
  return useQuery({ queryKey: ['projects'], queryFn: () => api.listProjects() });
}

export function useProject(id: number | null) {
  /** Fetch one project when a route id is available. */
  return useQuery({
    queryKey: ['project', id],
    enabled: id != null,
    queryFn: () => api.getProject(id as number),
  });
}

export function useQuickCreate() {
  /** Create a project from the quick-generation form. */
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: QuickCreateInput) => api.quickCreate(input),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
}

/** Project detail that polls while the pipeline is still running. */
export function useLiveProject(id: number | null) {
  return useQuery({
    queryKey: ['project', id],
    enabled: id != null,
    queryFn: () => api.getProject(id as number),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === 'completed' || status === 'failed' ? false : 1500;
    },
  });
}

export function useProjectBlocks(projectId: number | null) {
  /** Fetch the ordered blocks for a selected project. */
  return useQuery({
    queryKey: ['blocks', projectId],
    enabled: projectId != null,
    queryFn: () => api.listBlocks(projectId as number),
  });
}

export function useCreateProject() {
  /** Create a detailed project and refresh the project list. */
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: CreateProjectInput) => api.createProject(input),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
}

export function useDeleteProject() {
  /** Delete a project and invalidate its list entry. */
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteProject(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
}

export function useGenerateAll(projectId: number) {
  /** Queue complete generation and refresh project/block cache entries. */
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.generateAll(projectId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project', projectId] });
      qc.invalidateQueries({ queryKey: ['blocks', projectId] });
    },
  });
}

export function useCancel(projectId: number) {
  /** Request cancellation and refresh the project status. */
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.cancelProject(projectId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['project', projectId] }),
  });
}

export function useRerender(projectId: number) {
  /** Queue render-only regeneration for an existing project. */
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.rerender(projectId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project', projectId] });
    },
  });
}

export function useRegenerateBlockVisual(projectId: number) {
  /** Queue one block's visual regeneration and invalidate related queries. */
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (blockId: number) => api.regenerateBlockVisual(blockId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project', projectId] });
      qc.invalidateQueries({ queryKey: ['blocks', projectId] });
    },
  });
}

export function useRegenerateBlockAudio(projectId: number) {
  /** Queue one block's audio regeneration and invalidate related queries. */
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (blockId: number) => api.regenerateBlockAudio(blockId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project', projectId] });
      qc.invalidateQueries({ queryKey: ['blocks', projectId] });
    },
  });
}

export function useSpeakers(url?: string) {
  /** Fetch speaker choices for the VOICEVOX settings controls. */
  return useQuery({
    queryKey: ['speakers', url ?? 'default'],
    queryFn: () => api.speakers(url),
    retry: 0,
  });
}
