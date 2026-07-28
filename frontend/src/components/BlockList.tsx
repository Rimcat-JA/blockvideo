/** Ordered collection view for the generated blocks in a project. */
import type { BlockSummary } from '@/lib/types';
import { BlockItem } from './BlockItem';

/** Inputs required to render the block collection. */
interface BlockListProps {
  projectId: number;
  blocks: BlockSummary[];
}

export function BlockList({ projectId, blocks }: BlockListProps) {
  /** Show an empty-state message or one BlockItem per block. */
  if (blocks.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-slate-300 bg-slate-50 p-6 text-center text-sm text-slate-500">
        まだブロックがありません。プロジェクト詳細画面で「生成開始」を押してください。
      </div>
    );
  }
  return (
    <div className="space-y-3">
      {blocks.map((block) => (
        <BlockItem key={block.id} block={block} projectId={projectId} />
      ))}
    </div>
  );
}
