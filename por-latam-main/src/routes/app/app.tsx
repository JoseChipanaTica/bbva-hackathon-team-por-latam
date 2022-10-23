import * as React from "react";
import {
  createTheme,
  ThemeProvider,
  CssBaseline,
  Box,
  Toolbar,
  Container,
} from "@mui/material";

import { Outlet } from "react-router-dom";
import { AppDrawer } from "./components/app-drawer/app-drawer";
import { AppBar } from "./components/app-bar/app-bar";

// const mdTheme = createTheme({
//   palette: {
//     mode: "dark",
//   },
// });

export function AppPage() {
  const [open, setOpen] = React.useState(true);
  const toggleDrawer = () => {
    setOpen(!open);
  };
  // console.log("kiki");

  return (
    <Box sx={{ display: "flex" }}>
      <CssBaseline />
      <AppBar open={open} toggleDrawer={toggleDrawer} />
      <AppDrawer open={open} toggleDrawer={toggleDrawer} />
      <Box
        component="main"
        sx={{
          // backgroundColor: (theme) =>
          //   theme.palette.mode === "light"
          //     ? theme.palette.grey[100]
          //     : theme.palette.grey[900],
          backgroundColor: "transparent",
          flexGrow: 1,
          height: "100vh",
          overflow: "auto",
        }}
      >
        <Toolbar />
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
          <Outlet />
        </Container>
      </Box>
    </Box>
  );
}
