import * as React from "react";
import {
  styled,
  Toolbar,
  Typography,
  IconButton,
  Badge,
  Box,
  Avatar,
} from "@mui/material";

import MuiAppBar, { AppBarProps as MuiAppBarProps } from "@mui/material/AppBar";

import MenuIcon from "@mui/icons-material/Menu";
import NotificationsIcon from "@mui/icons-material/Notifications";

import { useLocation } from "react-router-dom";
import { APP_ROUTES } from "../../../constants";
import { KeyboardArrowDown, NotificationsOutlined } from "@mui/icons-material";

const drawerWidth: number = 240;

interface AppBarProps extends MuiAppBarProps {
  open?: boolean;
}

const StyledAppBar = styled(MuiAppBar, {
  shouldForwardProp: (prop) => prop !== "open",
})<AppBarProps>(({ theme, open }) => ({
  zIndex: theme.zIndex.drawer + 1,
  transition: theme.transitions.create(["width", "margin"], {
    easing: theme.transitions.easing.sharp,
    duration: theme.transitions.duration.leavingScreen,
  }),
  ...(open && {
    marginLeft: drawerWidth,
    width: `calc(100% - ${drawerWidth}px)`,
    transition: theme.transitions.create(["width", "margin"], {
      easing: theme.transitions.easing.sharp,
      duration: theme.transitions.duration.enteringScreen,
    }),
  }),
}));

interface CustomAppBarProps {
  open?: boolean;
  toggleDrawer: () => void;
}

export const AppBar = ({ open, toggleDrawer }: CustomAppBarProps) => {
  // const { pathname } = useLocation();
  // const fullRoute = pathname.split("/");
  // const currentRoute = APP_ROUTES[fullRoute[2] || ""];
  return (
    <StyledAppBar color="transparent" position="absolute" open={open}>
      <Toolbar
        sx={{
          pr: "24px", // keep right padding when drawer closed
        }}
      >
        <IconButton
          edge="start"
          color="inherit"
          aria-label="open drawer"
          onClick={toggleDrawer}
          sx={{
            transition: "opacity 0.5s",
            opacity: open ? 0 : 1,
            marginRight: "36px",
            ...(open && { display: "none" }),
          }}
        >
          <MenuIcon />
        </IconButton>
        <Box sx={{ flexGrow: 1 }} />
        {/* <Typography
          component="h1"
          variant="h6"
          color="inherit"
          noWrap
          sx={{ flexGrow: 1 }}
        >
          {currentRoute.name}
        </Typography> */}
        <Box sx={{ display: "flex" }}>
          <IconButton color="inherit">
            <Badge badgeContent={4} color="secondary">
              <NotificationsOutlined />
            </Badge>
          </IconButton>
          <Box sx={{ ml: 1 }} />
          <Box sx={{ display: "flex" }}>
            <Avatar>T</Avatar>
            <Box sx={{ ml: 1 }} />
            <Box sx={{ display: "flex", flexDirection: "column" }}>
              <Typography variant="body1">Team Por LATAM</Typography>
              <Typography variant="body2">Administrador</Typography>
            </Box>
            <Box sx={{ ml: 1 }} />
            <IconButton color="inherit">
              <KeyboardArrowDown />
            </IconButton>
          </Box>
        </Box>
      </Toolbar>
    </StyledAppBar>
  );
};
