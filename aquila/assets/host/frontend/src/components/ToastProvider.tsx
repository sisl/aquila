import {
  ReactNode,
  createContext,
  useCallback,
  useContext,
  useRef,
  useState
} from "react";
import { Alert, Snackbar } from "@mui/material";

type ToastSeverity = "success" | "error" | "info" | "warning";

type Toast = {
  id: number;
  message: string;
  severity: ToastSeverity;
};

type ToastContextValue = {
  success: (message: string) => void;
  error: (message: string) => void;
  info: (message: string) => void;
};

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const value = useContext(ToastContext);
  if (!value) {
    throw new Error("useToast must be used inside ToastProvider");
  }
  return value;
}

export function ToastProvider({ children }: { children: ReactNode }) {
  const [queue, setQueue] = useState<Toast[]>([]);
  const nextId = useRef(1);

  const push = useCallback((message: string, severity: ToastSeverity) => {
    setQueue((prev) => [...prev, { id: nextId.current++, message, severity }]);
  }, []);

  const value: ToastContextValue = {
    success: useCallback((message: string) => push(message, "success"), [push]),
    error: useCallback((message: string) => push(message, "error"), [push]),
    info: useCallback((message: string) => push(message, "info"), [push])
  };

  const current = queue[0] ?? null;
  const dismiss = () => setQueue((prev) => prev.slice(1));

  return (
    <ToastContext.Provider value={value}>
      {children}
      <Snackbar
        key={current?.id}
        open={current !== null}
        autoHideDuration={current?.severity === "error" ? 8000 : 4000}
        onClose={(_, reason) => {
          if (reason !== "clickaway") {
            dismiss();
          }
        }}
        anchorOrigin={{ vertical: "bottom", horizontal: "right" }}
      >
        {current ? (
          <Alert
            severity={current.severity}
            variant="filled"
            onClose={dismiss}
            sx={{ maxWidth: 480 }}
          >
            {current.message}
          </Alert>
        ) : undefined}
      </Snackbar>
    </ToastContext.Provider>
  );
}
