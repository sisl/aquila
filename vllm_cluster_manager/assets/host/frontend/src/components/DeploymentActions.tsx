import type { ReactNode } from "react";

import DeleteOutline from "@mui/icons-material/DeleteOutline";
import LinkOutlined from "@mui/icons-material/LinkOutlined";
import PauseCircleOutline from "@mui/icons-material/PauseCircleOutline";
import PlayArrowOutlined from "@mui/icons-material/PlayArrowOutlined";
import PlayCircleOutline from "@mui/icons-material/PlayCircleOutline";
import PushPin from "@mui/icons-material/PushPin";
import PushPinOutlined from "@mui/icons-material/PushPinOutlined";
import SettingsOutlined from "@mui/icons-material/SettingsOutlined";
import StopCircleOutlined from "@mui/icons-material/StopCircleOutlined";
import TerminalOutlined from "@mui/icons-material/TerminalOutlined";
import { Box, Button, Tooltip } from "@mui/material";

import type { Deployment } from "../services/api";
import { useToast } from "./ToastProvider";

type DeploymentActionsProps = {
  deployment: Deployment;
  isWarmNode?: (nodeId: number) => boolean;
  onSettings: () => void;
  onLogs: () => void;
  onEndpoint?: () => void;
  onPin?: (deployment: Deployment, pinned: boolean) => void;
  onPause?: () => void;
  onResume?: () => void;
  onStopClick: () => void;
  onRestart: () => void;
  onDeleteClick: () => void;
};

const ICON_SX = { fontSize: 16, opacity: 0.45, transition: "opacity 120ms ease" } as const;
const ICON_ERR_SX = { fontSize: 16, opacity: 0.55, transition: "opacity 120ms ease" } as const;

const BTN_SX = {
  minWidth: "unset",
  px: 0.75,
  py: 0.5,
  position: "relative",
  "@media (hover: hover)": {
    "&:hover .MuiSvgIcon-root": { opacity: 1 },
    "& .act-label": {
      position: "absolute",
      top: "calc(100% + 2px)",
      left: "50%",
      transform: "translateX(-50%)",
      fontSize: "0.55rem",
      lineHeight: 1,
      letterSpacing: "0.02em",
      opacity: 0,
      whiteSpace: "nowrap",
      pointerEvents: "none",
      transition: "opacity 100ms ease 80ms",
    },
    "&:hover .act-label": {
      opacity: 0.7,
      transition: "opacity 120ms ease-out",
    },
  },
  "@media (hover: none)": {
    "& .act-label": { display: "none" },
  },
} as const;

const WRAPPER_SX = {
  display: "flex",
  gap: 0.5,
  justifyContent: "flex-end",
  alignItems: "center",
} as const;

function ActionBtn({
  tooltip,
  label,
  icon,
  onClick,
  color,
}: {
  tooltip: string;
  label?: string;
  icon: ReactNode;
  onClick: () => void;
  color?: "primary" | "error";
}) {
  return (
    <Tooltip title={tooltip} enterDelay={2000}>
      <Button
        variant="text"
        size="small"
        color={color ?? "primary"}
        aria-label={tooltip}
        onClick={onClick}
        sx={BTN_SX}
      >
        {icon}
        <span className="act-label" aria-hidden>{label ?? tooltip}</span>
      </Button>
    </Tooltip>
  );
}

function canRestart(status: string) {
  return status === "stopped" || status === "expired" || status === "error";
}

export function DeploymentActions({
  deployment,
  isWarmNode,
  onSettings,
  onLogs,
  onEndpoint,
  onPin,
  onPause,
  onResume,
  onStopClick,
  onRestart,
  onDeleteClick,
}: DeploymentActionsProps) {
  const toast = useToast();
  const status = deployment.status;
  const warm = isWarmNode?.(deployment.node_id) ?? false;

  return (
    <Box sx={WRAPPER_SX}>
      <ActionBtn
        tooltip="Settings"
        icon={<SettingsOutlined sx={ICON_SX} />}
        onClick={onSettings}
      />

      <ActionBtn
        tooltip="Logs"
        icon={<TerminalOutlined sx={ICON_SX} />}
        onClick={onLogs}
      />

      {onEndpoint && (status === "running" || status === "paused_ram") && (
        <ActionBtn
          tooltip="Endpoint"
          icon={<LinkOutlined sx={ICON_SX} />}
          onClick={onEndpoint}
        />
      )}

      {onPin &&
        (status === "running" || status === "paused_ram") &&
        (status !== "running" || warm) && (
          <ActionBtn
            tooltip={deployment.pinned ? "Unpin" : "Pin"}
            icon={
              deployment.pinned
                ? <PushPin sx={ICON_SX} />
                : <PushPinOutlined sx={ICON_SX} />
            }
            onClick={() => onPin(deployment, !deployment.pinned)}
          />
        )}

      {onPause && status === "running" && warm && (
        <ActionBtn
          tooltip="Pause"
          icon={<PauseCircleOutline sx={ICON_SX} />}
          onClick={deployment.pinned
            ? () => toast.info("Cannot pause a pinned deployment. Unpin it first.")
            : onPause}
        />
      )}

      {onResume && status === "paused_ram" && (
        <ActionBtn
          tooltip="Resume"
          icon={<PlayCircleOutline sx={ICON_SX} />}
          onClick={onResume}
        />
      )}

      {(status === "running" || status === "loading" || status === "paused_ram") && (
        <ActionBtn
          tooltip="Stop"
          icon={<StopCircleOutlined sx={ICON_ERR_SX} />}
          onClick={onStopClick}
          color="error"
        />
      )}

      {canRestart(status) && (
        <ActionBtn
          tooltip="Start"
          icon={<PlayArrowOutlined sx={ICON_SX} />}
          onClick={onRestart}
        />
      )}

      {canRestart(status) && (
        <ActionBtn
          tooltip="Delete"
          icon={<DeleteOutline sx={ICON_ERR_SX} />}
          onClick={onDeleteClick}
          color="error"
        />
      )}
    </Box>
  );
}
