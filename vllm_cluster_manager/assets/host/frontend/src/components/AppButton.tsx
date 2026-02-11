import type { ButtonHTMLAttributes, ReactNode } from "react";
import clsx from "clsx";

type AppButtonVariant = "default" | "stop";

type AppButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: AppButtonVariant;
  children: ReactNode;
};

export function AppButton({
  variant = "default",
  className,
  children,
  ...rest
}: AppButtonProps) {
  return (
    <button className={clsx("app-button", `app-button--${variant}`, className)} {...rest}>
      {children}
    </button>
  );
}
