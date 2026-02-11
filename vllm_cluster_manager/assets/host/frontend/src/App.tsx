import { CssBaseline, GlobalStyles } from "@mui/material";
import { ThemeProvider } from "@mui/material/styles";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";

import { Dashboard } from "./pages/Dashboard";
import { theme } from "./styles/theme";

const queryClient = new QueryClient();

export default function App() {
  return (
    <ThemeProvider theme={theme}>
      <GlobalStyles
        styles={{
          ".MuiButton-root.MuiButton-outlined.MuiButton-outlinedInherit:hover, .MuiButton-root.MuiButton-outlined.MuiButton-colorInherit:hover, .MuiButton-root.MuiButton-outlined.MuiButton-textInherit:hover": {
            color: "#334155 !important",
            borderColor: "#334155 !important",
            backgroundColor: "transparent !important"
          }
        }}
      />
      <CssBaseline />
      <QueryClientProvider client={queryClient}>
        <Dashboard />
      </QueryClientProvider>
    </ThemeProvider>
  );
}
