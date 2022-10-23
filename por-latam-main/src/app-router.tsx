import { createBrowserRouter } from "react-router-dom";
import {
  AppError,
  AppPage,
  AuthenticatorPage,
  ClientsPage,
  Dashboard,
  EditClientPage,
  HomePage,
  LoginPage,
  MainErrorPage,
  RequestsPage,
  ServicesPage,
  SettingsPage,
  SignupPage,
  UsersPage,
} from "./routes";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <HomePage />,
    errorElement: <MainErrorPage />,
    children: [
      {
        path: "login",
        element: <LoginPage />,
      },
      {
        path: "signup",
        element: <SignupPage />,
      },
    ],
  },
  {
    path: "app",
    element: <AppPage />,
    children: [
      {
        errorElement: <AppError />,
        children: [
          {
            index: true,
            element: <Dashboard />,
          },
          { path: "auth", element: <AuthenticatorPage /> },
          {
            path: "clients",
            element: <ClientsPage />,
            children: [{ path: ":id", element: <EditClientPage /> }],
          },
          { path: "services", element: <ServicesPage /> },
          { path: "users", element: <UsersPage /> },
          { path: "requests", element: <RequestsPage /> },
          { path: "settings", element: <SettingsPage /> },
        ],
      },
    ],
  },
]);

/* 
export const APP_ROUTES: { [k: string]: Route } = {
  "": { name: "Dashboard", path: "", icon: DashboardIcon },
  auth: { name: "Autenticación", path: "auth", icon: DashboardIcon },
  services: { name: "Servicios", path: "services", icon: DashboardIcon },
  users: { name: "Usuarios", path: "users", icon: DashboardIcon },
  requests: { name: "Solicitudes", path: "requests", icon: DashboardIcon },
  settings: { name: "Ajustes", path: "settings", icon: DashboardIcon },
};

*/
