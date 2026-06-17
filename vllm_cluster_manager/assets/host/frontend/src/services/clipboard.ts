export async function copyToClipboard(text: string): Promise<void> {
  // 1. Modern Clipboard API — only available on secure contexts (HTTPS / localhost).
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Permission denied — fall through to legacy path.
    }
  }

  // 2. Legacy execCommand fallback for plain-HTTP origins.
  //    Use a *visible* offscreen textarea (opacity:0 / display:none can cause
  //    Safari to skip the selection) and verify execCommand actually succeeded.
  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  Object.assign(textarea.style, {
    position: "fixed",
    left: "-9999px",
    top: "-9999px",
  } as CSSStyleDeclaration);
  document.body.appendChild(textarea);

  textarea.focus({ preventScroll: true });
  textarea.select();

  let ok = false;
  try {
    ok = document.execCommand("copy");
  } finally {
    document.body.removeChild(textarea);
  }

  if (!ok) {
    throw new Error("Clipboard write failed");
  }
}
