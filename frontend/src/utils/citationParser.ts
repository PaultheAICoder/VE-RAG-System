/**
 * Citation parser utility for transforming raw SourceId citations
 * into numbered footnotes.
 *
 * Backend generates: [SourceId: 550e8400-e29b-41d4-a716-446655440000:0]
 * This transforms to: [1], [2], etc.
 */

import type { SourceInfo } from '../types';

// Pattern matching backend SOURCEID_PATTERN from rag_service.py
const CITATION_PATTERN = /\[SourceId:\s*([a-f0-9-]{36}:\d+)\]/gi;

export interface CitationMap {
  [sourceId: string]: number; // sourceId -> footnote number
}

export interface ParsedContent {
  text: string; // Content with [1], [2] footnotes
  citationMap: CitationMap; // Map for linking footnotes to sources
}

/**
 * Parse message content and replace SourceId citations with numbered footnotes.
 *
 * @param content - Raw message content with [SourceId: ...] citations
 * @param sources - Optional array of source info for validation
 * @returns Parsed content with footnotes and citation map
 */
export function parseCitations(
  content: string,
  _sources?: SourceInfo[]
): ParsedContent {
  const citationMap: CitationMap = {};
  let footnoteNum = 1;

  // Replace [SourceId: uuid:index] with [N]
  const text = content.replace(CITATION_PATTERN, (_match, sourceId: string) => {
    if (!citationMap[sourceId]) {
      citationMap[sourceId] = footnoteNum++;
    }
    return `[${citationMap[sourceId]}]`;
  });

  return { text, citationMap };
}

/**
 * Get footnote number for a source ID.
 *
 * @param sourceId - The source ID to look up
 * @param citationMap - The citation map from parseCitations
 * @returns Footnote number or undefined if not cited
 */
export function getFootnoteNumber(
  sourceId: string,
  citationMap: CitationMap
): number | undefined {
  return citationMap[sourceId];
}
