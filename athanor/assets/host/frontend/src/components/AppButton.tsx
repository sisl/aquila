import type { CSSProperties, MouseEventHandler, ReactNode } from "react";
import { Button } from "@mui/material";

type AppButtonVariant = "default" | "stop";

type AppButtonProps = {
  variant?: AppButtonVariant;
  // Call sites pass "app-button--small" for the compact size; mapped to MUI size.
  className?: string;
  // Quiet text-only button for row-level actions: muted label, no border,
  // sharpens on hover (styled once in styles/theme.ts MuiButton `text`).
  ghost?: boolean;
  type?: "button" | "submit" | "reset";
  disabled?: boolean;
  onClick?: MouseEventHandler<HTMLButtonElement>;
  style?: CSSProperties;
  children: ReactNode;
};

// Thin wrapper over MUI Button so the whole app uses one button system
// (hover/focus/disabled states defined once in styles/theme.ts).
export function AppButton({
  variant = "default",
  className,
  ghost = false,
  type = "button",
  disabled,
  onClick,
  style,
  children
}: AppButtonProps) {
  const small = className?.includes("app-button--small") ?? false;
  return (
    <Button
      variant={ghost ? "text" : "outlined"}
      color={variant === "stop" ? "error" : "primary"}
      size={small ? "small" : "medium"}
      type={type}
      disabled={disabled}
      onClick={onClick}
      style={style}
    >
      {children}
    </Button>
  );
}
