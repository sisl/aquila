import {
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Box,
  Typography
} from "@mui/material";

import type { Node } from "../services/api";

type NodeTableProps = {
  nodes: Node[];
};

const toGb = (mb?: number) => {
  if (mb === undefined) {
    return null;
  }
  return (mb / 1024).toFixed(1);
};

export function NodeTable({ nodes }: NodeTableProps) {
  return (
    <TableContainer component={Box} sx={{ minHeight: 200 }}>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Hostname</TableCell>
            <TableCell>IP Address</TableCell>
            <TableCell>Client Port</TableCell>
            <TableCell>GPU Compute Utilization</TableCell>
            <TableCell>Status</TableCell>
            <TableCell align="right">Last Heartbeat</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {nodes.map((node) => (
            <TableRow key={node.id} hover>
              <TableCell>{node.hostname}</TableCell>
              <TableCell>{node.ip_address}</TableCell>
              <TableCell>{node.port ?? "-"}</TableCell>
              <TableCell>
                {node.gpu_usage && node.gpu_usage.length > 0 ? (
                  <Box sx={{ display: "flex", flexDirection: "column", gap: 0.5 }}>
                    {node.gpu_usage.map((gpu) => {
                      const used = toGb(gpu.memory_used_mb);
                      const total = toGb(gpu.memory_total_mb);
                      const memLabel =
                        used && total ? `${used}/${total} GB` : "unknown";
                      const sourceLabel =
                        gpu.source === "unified" ? " (unified memory)" : "";
                      return (
                        <Typography key={`${node.id}-gpu-${gpu.index}`} variant="body2">
                          GPU{gpu.index} {gpu.utilization ?? "?"}% ({memLabel})
                          {sourceLabel}
                        </Typography>
                      );
                    })}
                  </Box>
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    -
                  </Typography>
                )}
              </TableCell>
              <TableCell>
                <Chip
                  label={node.status}
                  size="small"
                  color={node.status === "healthy" ? "success" : "default"}
                />
              </TableCell>
              <TableCell align="right">{node.last_heartbeat_at ?? "-"}</TableCell>
            </TableRow>
          ))}
          {nodes.length === 0 && (
            <TableRow>
              <TableCell colSpan={6}>No registered nodes yet.</TableCell>
            </TableRow>
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
