import { useState } from "react";

export type ColumnDef = {
  key: string;
  label: string;
  alwaysVisible?: boolean;
  compactHidden?: boolean;
};

function loadHidden(storageKey: string, validKeys: Set<string>): Set<string> {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return new Set();
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return new Set();
    return new Set(parsed.filter((k): k is string => typeof k === "string" && validKeys.has(k)));
  } catch {
    return new Set();
  }
}

function persist(storageKey: string, hidden: Set<string>) {
  if (hidden.size === 0) {
    localStorage.removeItem(storageKey);
  } else {
    localStorage.setItem(storageKey, JSON.stringify([...hidden]));
  }
}

export function useColumnVisibility(
  storageKey: string,
  columns: ColumnDef[],
  compact: boolean
) {
  const validKeys = new Set(columns.filter((c) => !c.alwaysVisible).map((c) => c.key));
  const [userHidden, setUserHidden] = useState(() => loadHidden(storageKey, validKeys));

  const visibleKeys = new Set<string>();
  for (const col of columns) {
    if (col.alwaysVisible) {
      visibleKeys.add(col.key);
      continue;
    }
    if (compact && col.compactHidden) continue;
    if (!userHidden.has(col.key)) visibleKeys.add(col.key);
  }

  const toggle = (key: string) => {
    setUserHidden((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      persist(storageKey, next);
      return next;
    });
  };

  const reset = () => {
    setUserHidden(new Set());
    localStorage.removeItem(storageKey);
  };

  return {
    visibleKeys,
    userHidden,
    toggle,
    reset,
    isCustomized: userHidden.size > 0,
    columns,
  };
}
