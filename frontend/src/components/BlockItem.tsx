/** Detailed block card with artifacts, stage state, and regeneration actions. */
import type { BlockSummary, PartialGenerationKind } from '@/lib/types';
import { StatusBadge } from './StatusBadge';

/** Block card inputs, including the parent project used by mutations. */
interface BlockItemProps {
  block: BlockSummary;
  disabled?: boolean;
  onGenerate: (kind: PartialGenerationKind, blockIndex: number) => void;
}

function formatMs(ms: number | null | undefined): string {
  /** Format nullable milliseconds using the Japanese UI unit. */
  if (ms == null) return '-';
  return `${(ms / 1000).toFixed(2)}秒`;
}

export function BlockItem({ block, disabled, onGenerate }: BlockItemProps) {
  /** Render block metadata and controls for partial regeneration. */

  return (
    <article className="space-y-3 rounded-lg border border-slate-200 bg-white p-4">
      <header className="flex items-start justify-between gap-2">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-semibold text-slate-700">#{block.index}</span>
            <span className="badge bg-slate-100 text-slate-700">{block.visual_type}</span>
          </div>
          <p className="mt-2 text-sm text-slate-700">{block.source_text}</p>
        </div>
        {block.image_url && (
          <img
            src={block.image_url}
            alt={`block-${block.index} image`}
            className="h-20 w-32 rounded border border-slate-200 object-cover"
          />
        )}
      </header>

      <details className="text-xs text-slate-600">
        <summary className="cursor-pointer">読み上げテキスト</summary>
        <p className="mt-1 rounded bg-slate-50 p-2">{block.tts_text}</p>
      </details>

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <StatusBadge status={block.status_split} />
        分割
        <StatusBadge status={block.status_visual_plan} />
        プラン
        <StatusBadge status={block.status_image} />
        画像
        <StatusBadge status={block.status_audio} />
        音声
        <StatusBadge status={block.status_render} />
        レンダ
        <span className="ml-auto text-slate-500">
          音声 {formatMs(block.duration_ms)} / 表示 {formatMs(block.display_duration_ms)}
        </span>
      </div>

      {block.error_message && (
        <div className="rounded border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-700">
          {block.error_message}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        {block.audio_url && (
          <audio controls src={block.audio_url} className="h-8 w-72" />
        )}
        <button
          type="button"
          className="btn-secondary"
          onClick={() => onGenerate('block_visual', block.index)}
          disabled={disabled}
        >
          画像だけ再生成
        </button>
        <button
          type="button"
          className="btn-secondary"
          onClick={() => onGenerate('block_audio', block.index)}
          disabled={disabled}
        >
          音声だけ再生成
        </button>
        <button
          type="button"
          className="btn-secondary"
          onClick={() => onGenerate('rerender', block.index)}
          disabled={disabled}
        >
          このブロックから再レンダリング
        </button>
        {block.visual_plan && (
          <details className="ml-auto text-xs">
            <summary className="cursor-pointer text-slate-500">visual_plan</summary>
            <pre className="mt-1 max-h-40 overflow-auto rounded bg-slate-50 p-2 text-[10px]">
              {JSON.stringify(block.visual_plan, null, 2)}
            </pre>
          </details>
        )}
      </div>
    </article>
  );
}
