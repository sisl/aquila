export async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
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

  textarea.focus({ preventScroll: true });
  textarea.select();
  textarea.setSelectionRange(0, textarea.value.length);

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
