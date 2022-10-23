import { ClientsTable } from "./table";
const TEMPLATE_DATA = Array.from(new Array(375), (_, i) => ({
  id: i + 1,
  name: "Nombre Apellido",
  info: "xxx-xxx-xxx-xxx",
  contact: "client@app.com",
  status: ["Activo", "Pendiente", "Pausado", "Inactivo"][i % 4],
}));

export const ClientsPage = () => {
  return <ClientsTable data={TEMPLATE_DATA} />;
};
