import { useState } from "react";

export type ColumnDef = {
  key: string;
  label: string;
  alwaysVisible?: boolean;
  defaultHidden?: boolean;
};

function defaultHiddenSet(columns: ColumnDef[]): Set<string> {
  return new Set(columns.filter((c) => c.defaultHidden && !c.alwaysVisible).map((c) => c.key));
}

function loadHidden(storageKey: string, validKeys: Set<string>, columns: ColumnDef[]): Set<string> {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return defaultHiddenSet(columns);
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return defaultHiddenSet(columns);
    return new Set(parsed.filter((k): k is string => typeof k === "string" && validKeys.has(k)));
  } catch {
    return defaultHiddenSet(columns);
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
  columns: ColumnDef[]
) {
  const validKeys = new Set(columns.filter((c) => !c.alwaysVisible).map((c) => c.key));
  const [userHidden, setUserHidden] = useState(() => loadHidden(storageKey, validKeys, columns));

  const visibleKeys = new Set<string>();
  for (const col of columns) {
    if (col.alwaysVisible) {
      visibleKeys.add(col.key);
      continue;
    }
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

  const defaults = defaultHiddenSet(columns);

  const reset = () => {
    setUserHidden(defaults);
    persist(storageKey, defaults);
  };

  const isCustomized =
    userHidden.size !== defaults.size ||
    [...userHidden].some((k) => !defaults.has(k));

  return {
    visibleKeys,
    userHidden,
    toggle,
    reset,
    isCustomized,
    columns,
  };
}
