import { ReactNode } from "react";
import { Button, Typography } from "@mui/material";

import { AppButton } from "./AppButton";
import { AppDialog } from "./AppDialog";

type ConfirmDialogProps = {
  open: boolean;
  title: string;
  body?: ReactNode;
  confirmLabel?: string;
  danger?: boolean;
  onConfirm: () => void;
  onCancel: () => void;
};

export function ConfirmDialog({
  open,
  title,
  body,
  confirmLabel = "Confirm",
  danger = false,
  onConfirm,
  onCancel
}: ConfirmDialogProps) {
  return (
    <AppDialog
      open={open}
      onClose={onCancel}
      title={title}
      maxWidth="xs"
      actions={
        <>
          <AppButton type="button" onClick={onCancel}>
            Cancel
          </AppButton>
          <Button
            variant="contained"
            color={danger ? "error" : "primary"}
            onClick={onConfirm}
          >
            {confirmLabel}
          </Button>
        </>
      }
    >
      {typeof body === "string" ? (
        <Typography variant="body2">{body}</Typography>
      ) : (
        body ?? <Typography variant="body2">Are you sure?</Typography>
      )}
    </AppDialog>
  );
}
