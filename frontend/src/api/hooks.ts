import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/api/client';
import type { CreateProjectInput, QuickCreateInput } from '@/lib/types';

export function useProjects() {
  return useQuery({ queryKey: ['projects'], queryFn: () => api.listProjects() });
}

export function useProject(id: number | null) {
  return useQuery({
    queryKey: ['project', id],
    enabled: id != null,
    queryFn: () => api.getProject(id as number),
  });
}

export function useQuickCreate() {
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
  return useQuery({
    queryKey: ['blocks', projectId],
    enabled: projectId != null,
    queryFn: () => api.listBlocks(projectId as number),
  });
}

export function useCreateProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (input: CreateProjectInput) => api.createProject(input),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
}

export function useDeleteProject() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: number) => api.deleteProject(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['projects'] }),
  });
}

export function useGenerateAll(projectId: number) {
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
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.cancelProject(projectId),
    onSuccess: () => qc.invalidateQueries({ queryKey: ['project', projectId] }),
  });
}

export function useRerender(projectId: number) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api.rerender(projectId),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['project', projectId] });
    },
  });
}

export function useRegenerateBlockVisual(projectId: number) {
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
  return useQuery({
    queryKey: ['speakers', url ?? 'default'],
    queryFn: () => api.speakers(url),
    retry: 0,
  });
}