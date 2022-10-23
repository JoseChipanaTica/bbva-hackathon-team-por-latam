import React from "react";
import {
  Box,
  Typography,
  Toolbar,
  IconButton,
  Button,
  AppBar,
} from "@mui/material";
import { Menu } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { AppBarMenu } from "./app-bar-menu";

const HOME_ROUTES = [
  { name: "Home", path: "/" },
  { name: "App", path: "app" },
  { name: "Log in", path: "login" },
  { name: "Signup", path: "signup" },
];

const MENU_OPTIONS = [
  { name: "Servicio 1", id: "menu-1", items: [] },
  { name: "Servicio 2", id: "menu-2", items: [] },
  { name: "Servicio 3", id: "menu-3", items: [] },
  { name: "Servicio 4", id: "menu-4", items: [] },
  { name: "Servicio 5", id: "menu-5", items: [] },
];

export const HomeAppBar = (props: any) => {
  const { handleDrawerToggle } = props;
  const navigate = useNavigate();
  const [anchorEl, setAnchorEl] = React.useState<null | HTMLElement>(null);
  const [selectedMenu, setSelectedMenu] = React.useState<any>(null);
  const open = Boolean(anchorEl);
  const handleClick = (element: HTMLElement, menu: any) => {
    setAnchorEl(element);
    setSelectedMenu(menu);
  };
  const handleClose = () => {
    setAnchorEl(null);
    setSelectedMenu(null);
  };
  return (
    <>
      <AppBar component="nav">
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: "none" } }}
          >
            <Menu />
          </IconButton>
          <Typography
            variant="h6"
            component="div"
            sx={{ flexGrow: 1, display: { xs: "none", sm: "block" } }}
          >
            AUTENTICANOS
          </Typography>
          <Box sx={{ display: { xs: "none", sm: "block" } }}>
            {MENU_OPTIONS.map((menu) => {
              return (
                <Box key={menu.name} sx={{ display: "inline", px: 2 }}>
                  <Button
                    onClick={(e) => {
                      handleClick(e.currentTarget, menu);
                    }}
                    sx={{ color: "#fff" }}
                    aria-controls={open ? menu.id : undefined}
                    aria-haspopup="true"
                    aria-expanded={open ? "true" : undefined}
                  >
                    {menu.name}
                  </Button>
                </Box>
              );
            })}
            <Button
              onClick={() => {
                navigate("app");
              }}
              sx={{ color: "#fff" }}
            >
              App
            </Button>
          </Box>
        </Toolbar>
      </AppBar>
      <AppBarMenu
        handleClose={handleClose}
        menuId={selectedMenu && selectedMenu.id}
        anchorEl={anchorEl}
      />
    </>
  );
};

/* 
{HOME_ROUTES.map((item) => (
            <Button
              onClick={() => {
                navigate(item.path);
              }}
              key={item.path}
              sx={{ color: "#fff" }}
            >
              {item.name}
            </Button>
          ))}
*/
