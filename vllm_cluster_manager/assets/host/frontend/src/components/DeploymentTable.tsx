import {
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Box
} from "@mui/material";

import type { Deployment } from "../services/api";
import { AppButton } from "./AppButton";

type DeploymentTableProps = {
  deployments: Deployment[];
  onStop: (deploymentId: number) => void;
  onDelete: (deploymentId: number) => void;
  onLogs: (deploymentId: number) => void;
  onSettings: (deployment: Deployment) => void;
  nodeNameById: Record<number, string>;
};

export function DeploymentTable({
  deployments,
  onStop,
  onDelete,
  onLogs,
  onSettings,
  nodeNameById
}: DeploymentTableProps) {
  const formatArgs = (args?: string[]) => {
    if (!args || args.length === 0) {
      return [];
    }
    const rows: string[] = [];
    let i = 0;
    while (i < args.length) {
      const token = String(args[i]);
      const next = args[i + 1] ? String(args[i + 1]) : "";
      const isFlag = token.startsWith("-");
      const hasValue = next && !next.startsWith("-");
      if (isFlag && hasValue) {
        rows.push(`${token} ${next}`);
        i += 2;
      } else {
        rows.push(token);
        i += 1;
      }
    }
    return rows;
  };

  return (
    <TableContainer component={Box} sx={{ minHeight: 200 }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Model</TableCell>
            <TableCell>Node</TableCell>
            <TableCell>Port</TableCell>
            <TableCell>vLLM</TableCell>
            <TableCell>GPU Fraction</TableCell>
            <TableCell>GPUs</TableCell>
            <TableCell>Args</TableCell>
            <TableCell>Status</TableCell>
            <TableCell align="right">Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {deployments.map((deployment) => {
            const argRows = formatArgs(deployment.extra_args);
            return (
              <TableRow key={deployment.id} hover>
                <TableCell>{deployment.model_name}</TableCell>
                <TableCell>{nodeNameById[deployment.node_id] ?? deployment.node_id}</TableCell>
                <TableCell>{deployment.port}</TableCell>
                <TableCell>
                  {deployment.vllm_version || "-"}
                  {deployment.extra_packages && deployment.extra_packages.length > 0 && (
                    <Chip label={`+${deployment.extra_packages.length} pkg`} size="small" sx={{ ml: 0.5 }} />
                  )}
                </TableCell>
                <TableCell>{deployment.gpu_memory_fraction}</TableCell>
                <TableCell>
                  {deployment.gpu_ids && deployment.gpu_ids.length > 0
                    ? deployment.gpu_ids.join(", ")
                    : "-"}
                </TableCell>
                <TableCell>
                  {argRows.length > 0 ? (
                    <Box
                      sx={{
                        display: "flex",
                        flexDirection: "column",
                        gap: 0.25,
                        fontFamily: "SFMono-Regular, ui-monospace, monospace",
                        fontSize: "0.75rem"
                      }}
                    >
                      {argRows.map((row, index) => (
                        <Box key={`${deployment.id}-arg-${index}`}>{row}</Box>
                      ))}
                    </Box>
                  ) : (
                    "-"
                  )}
                </TableCell>
                <TableCell>
                  <Chip
                    label={deployment.status}
                    size="small"
                    color={deployment.status === "running" ? "success" : "default"}
                  />
                </TableCell>
                <TableCell align="right">
                  <Box sx={{ display: "flex", gap: 1, justifyContent: "flex-end" }}>
                    <AppButton
                      type="button"
                      className="app-button--small"
                      onClick={() => onSettings(deployment)}
                    >
                      Settings
                    </AppButton>
                    <AppButton
                      type="button"
                      className="app-button--small"
                      onClick={() => onLogs(deployment.id)}
                    >
                      Logs
                    </AppButton>
                    {(deployment.status === "running" || deployment.status === "loading") && (
                      <AppButton
                        type="button"
                        variant="stop"
                        className="app-button--small"
                        onClick={() => onStop(deployment.id)}
                      >
                        Stop
                      </AppButton>
                    )}
                    {(deployment.status === "stopped" || deployment.status === "error") && (
                      <AppButton
                        type="button"
                        className="app-button--small"
                        onClick={() => onDelete(deployment.id)}
                      >
                        Delete
                      </AppButton>
                    )}
                  </Box>
                </TableCell>
              </TableRow>
            );
          })}
          {deployments.length === 0 && (
            <TableRow>
              <TableCell colSpan={9}>No deployments yet.</TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
