import { useEffect, useRef, useState } from "react";

import ContentCopyIcon from "@mui/icons-material/ContentCopy";
import { copyToClipboard } from "../services/clipboard";
import {
  Box,
  Chip,
  IconButton,
  Stack,
  Tooltip,
  Typography
} from "@mui/material";
import { useQuery } from "@tanstack/react-query";

import {
  createApiKey,
  fetchApiKeys,
  fetchSettings,
  gatewayBaseUrl
} from "../services/api";
import type { Deployment, Node } from "../services/api";
import { AppButton } from "./AppButton";
import { AppDialog } from "./AppDialog";
import { SectionLabel } from "./SectionLabel";
import { useToast } from "./ToastProvider";

type EndpointDialogProps = {
  deployment: Deployment | null;
  node: Node | null;
  gatewayEnabled?: boolean;
  open: boolean;
  onClose: () => void;
};

type UrlKind = "gateway" | "direct";

function servedName(deployment: Deployment): string {
  const served = deployment.engine_args?.served_model_name;
  return typeof served === "string" && served ? served : deployment.model_name;
}

function pythonSnippet(baseUrl: string, model: string, apiKey: string): string {
  return [
    "from openai import OpenAI",
    "",
    `client = OpenAI(base_url="${baseUrl}", api_key="${apiKey}")`,
    "resp = client.chat.completions.create(",
    `    model="${model}",`,
    '    messages=[{"role": "user", "content": "Hello"}],',
    ")",
    "print(resp.choices[0].message.content)"
  ].join("\n");
}

function curlSnippet(baseUrl: string, model: string, apiKey: string): string {
  const lines = [
    `curl ${baseUrl}/chat/completions \\`,
    '  -H "Content-Type: application/json" \\'
  ];
  if (apiKey !== "not-needed") {
    lines.push(`  -H "Authorization: Bearer ${apiKey}" \\`);
  }
  lines.push(
    `  -d '{"model": "${model}", "messages": [{"role": "user", "content": "Hello"}]}'`
  );
  return lines.join("\n");
}

type EndpointBlockProps = {
  label: string;
  content: (baseUrl: string) => string;
  gatewayUrl: string | null;
  directUrl: string | null;
  kind: UrlKind;
  onKindChange: (kind: UrlKind) => void;
  onCopy: (label: string, text: string) => void;
};

function UrlKindOption({
  label,
  active,
  onClick
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <Typography
      component="button"
      variant="caption"
      onClick={onClick}
      sx={{
        all: "unset",
        cursor: "pointer",
        fontSize: "0.7rem",
        letterSpacing: "0.04em",
        px: 0.5,
        fontWeight: active ? 600 : 400,
        color: active ? "text.primary" : "text.secondary",
        transition: "color 120ms ease",
        "&:hover": { color: "text.primary" },
        "&:focus-visible": { outline: "2px solid var(--accent)", outlineOffset: 2 }
      }}
    >
      {label}
    </Typography>
  );
}

function EndpointBlock({ label, content, gatewayUrl, directUrl, kind, onKindChange, onCopy }: EndpointBlockProps) {
  const baseUrl =
    kind === "direct" && directUrl ? directUrl : gatewayUrl ?? directUrl ?? "";
  const text = content(baseUrl);

  return (
    <Box>
      <SectionLabel sx={{ mb: 0.5 }}>{label}</SectionLabel>
      <Box
        sx={{
          border: "1px solid var(--line)",
          borderRadius: "var(--radius)"
        }}
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "flex-end",
            gap: 0.25,
            pt: 0.5,
            pr: 0.75,
            mb: -1.25
          }}
        >
          {directUrl && gatewayUrl && (
            <>
              <UrlKindOption
                label="gateway"
                active={kind === "gateway"}
                onClick={() => onKindChange("gateway")}
              />
              <Typography variant="caption" className="muted" sx={{ opacity: 0.5 }}>
                /
              </Typography>
              <UrlKindOption
                label="direct"
                active={kind === "direct"}
                onClick={() => onKindChange("direct")}
              />
            </>
          )}
          <Tooltip title="Copy" enterDelay={500}>
            <IconButton
              size="small"
              aria-label={`Copy ${label.toLowerCase()}`}
              onClick={() => onCopy(label, text)}
              sx={{ ml: 0.25 }}
            >
              <ContentCopyIcon sx={{ fontSize: 14 }} />
            </IconButton>
          </Tooltip>
        </Box>
        <Box className="code-area" sx={{ overflowX: "auto", px: 1.5, pb: 1.5 }}>
          {text}
        </Box>
      </Box>
    </Box>
  );
}

export function EndpointDialog({
  deployment,
  node,
  gatewayEnabled = true,
  open,
  onClose
}: EndpointDialogProps) {
  const toast = useToast();
  const [urlKind, setUrlKind] = useState<UrlKind>("gateway");
  const [tempKey, setTempKey] = useState<string | null>(null);
  const creatingRef = useRef(false);

  const settingsQuery = useQuery({
    queryKey: ["settings"],
    queryFn: fetchSettings,
    enabled: open
  });

  const apiKeysQuery = useQuery({
    queryKey: ["api-keys"],
    queryFn: fetchApiKeys,
    enabled: open
  });

  const ttl = settingsQuery.data?.temp_api_key_ttl_seconds ?? 300;
  const hasPermanentKeys = (apiKeysQuery.data ?? []).some((k) => k.expires_at === null);

  useEffect(() => {
    if (!open) {
      setTempKey(null);
      creatingRef.current = false;
      return;
    }
    if (!hasPermanentKeys || ttl <= 0 || creatingRef.current || tempKey) return;
    creatingRef.current = true;
    createApiKey("snippet (temp)", ttl)
      .then((result) => setTempKey(result.key))
      .catch(() => setTempKey(null))
      .finally(() => { creatingRef.current = false; });
  }, [open, hasPermanentKeys, ttl, tempKey]);

  if (!deployment) {
    return null;
  }

  const model = servedName(deployment);
  const gatewayUrl = gatewayEnabled ? gatewayBaseUrl() : null;
  const directUrl = node ? `http://${node.ip_address}:${deployment.port}/v1` : null;
  const loraNames = (deployment.lora_modules ?? [])
    .map((module) => module.name)
    .filter(Boolean);

  const apiKey = tempKey ?? "not-needed";
  const ttlMinutes = Math.round(ttl / 60);

  const copy = async (label: string, text: string) => {
    try {
      await copyToClipboard(text);
      toast.success(`${label} copied.`);
    } catch {
      toast.error("Clipboard unavailable.");
    }
  };

  return (
    <AppDialog
      open={open}
      onClose={onClose}
      title={
        <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 1 }}>
          Endpoint — {model}
          {loraNames.length > 0 && (
            <Chip label={`also serves: ${loraNames.join(", ")}`} size="small" />
          )}
        </Box>
      }
      meta={
        gatewayEnabled
          ? "The gateway URL is stable across node moves; the direct URL skips one hop."
          : "The gateway is disabled in Settings — direct node URLs only."
      }
      actions={
        <AppButton type="button" onClick={onClose}>
          Close
        </AppButton>
      }
    >
      <Stack spacing={2}>
        <EndpointBlock
          label="Base URL"
          content={(baseUrl) => baseUrl}
          gatewayUrl={gatewayUrl}
          directUrl={directUrl}
          kind={gatewayUrl ? urlKind : "direct"}
          onKindChange={setUrlKind}
          onCopy={copy}
        />
        <EndpointBlock
          label="Python (openai client)"
          content={(baseUrl) => pythonSnippet(baseUrl, model, apiKey)}
          gatewayUrl={gatewayUrl}
          directUrl={directUrl}
          kind={gatewayUrl ? urlKind : "direct"}
          onKindChange={setUrlKind}
          onCopy={copy}
        />
        <EndpointBlock
          label="curl"
          content={(baseUrl) => curlSnippet(baseUrl, model, apiKey)}
          gatewayUrl={gatewayUrl}
          directUrl={directUrl}
          kind={gatewayUrl ? urlKind : "direct"}
          onKindChange={setUrlKind}
          onCopy={copy}
        />
        {tempKey && (
          <Typography variant="caption" className="muted">
            Using a temporary API key valid for {ttlMinutes} min.
          </Typography>
        )}
      </Stack>
    </AppDialog>
  );
}
