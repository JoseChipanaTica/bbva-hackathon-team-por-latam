import {
  Box,
  Typography,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Drawer,
  Toolbar,
  IconButton,
  Button,
  AppBar,
} from "@mui/material";
import { Menu } from "@mui/icons-material";
import { useState } from "react";
import { HOME_DRAWER_WIDTH } from "../constants";
import { Outlet, useNavigate } from "react-router-dom";
import { HomeAppBar } from "./components/home-app-bar";
import { LoginPage } from "./login";

const HOME_ROUTES = [
  { name: "Home", path: "/" },
  { name: "App", path: "app" },
  { name: "Log in", path: "login" },
  { name: "Signup", path: "signup" },
];

export const HomePage = () => {
  return <LoginPage />;
};
