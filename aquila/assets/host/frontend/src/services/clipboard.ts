export async function copyToClipboard(text: string): Promise<void> {
  if (window.isSecureContext && navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text);
      return;
    } catch {
      // Permission denied or not allowed — fall through to legacy path.
    }
  }

  const textarea = document.createElement("textarea");
  textarea.value = text;
  textarea.setAttribute("readonly", "");
  Object.assign(textarea.style, {
    position: "fixed",
    left: "0",
    top: "0",
    width: "1px",
    height: "1px",
    padding: "0",
    border: "none",
    outline: "none",
    boxShadow: "none",
    background: "transparent",
    clip: "rect(0, 0, 0, 0)",
  });
  document.body.appendChild(textarea);

  // Temporarily suppress focusin so MUI Dialog's focus-trap doesn't steal
  // focus back from the textarea before execCommand can read the selection.
  const trap = (e: Event) => e.stopImmediatePropagation();
  document.addEventListener("focusin", trap, true);

  textarea.focus({ preventScroll: true });
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);

  let ok = false;
  try {
    ok = document.execCommand("copy");
  } finally {
    document.removeEventListener("focusin", trap, true);
    document.body.removeChild(textarea);
  }

  if (!ok) {
    throw new Error("Clipboard write failed");
  }
}
