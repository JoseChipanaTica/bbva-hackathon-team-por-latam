import { SvgIconTypeMap } from "@mui/material";
import { OverridableComponent } from "@mui/material/OverridableComponent";
import {
  Dashboard as DashboardIcon,
  Key,
  RoomService,
  PeopleAlt,
  InsertDriveFile,
  Settings,
  EmojiPeople,
} from "@mui/icons-material";

export const HOME_DRAWER_WIDTH = 240;
export const APP_DRAWER_WIDTH = 240;

interface Route {
  name: string;
  path?: string;
  children?: { [k: string]: Route };
  icon?: OverridableComponent<SvgIconTypeMap>;
}

export const APP_ROUTES: { [k: string]: Route } = {
  "": { name: "Dashboard", path: "", icon: DashboardIcon },
  auth: { name: "Autenticación", path: "auth", icon: Key },
  clients: { name: "Clientes", path: "clients", icon: EmojiPeople },
  // services: { name: "Servicios", path: "services", icon: RoomService },
  users: { name: "Usuarios", path: "users", icon: PeopleAlt },
  requests: { name: "Solicitudes", path: "requests", icon: InsertDriveFile },
  settings: { name: "Ajustes", path: "settings", icon: Settings },
};

export const LIST_ITEMS = Object.keys(APP_ROUTES).map(
  (routeName) => APP_ROUTES[routeName]
);
