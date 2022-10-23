import { useState } from "react";
import {
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TablePagination,
  TableRow,
  TextField,
  Box,
  Button,
  Typography,
  Divider,
  Grid,
  GlobalStyles,
} from "@mui/material";

import {
  Edit,
  Search as SearchIcon,
  Add as AddIcon,
  DeleteOutline,
} from "@mui/icons-material";
import { Outlet, useNavigate } from "react-router-dom";
import { customPalette } from "../../../theme";
import { PopupMessage } from "../components/popup-message";
// import { ProductView } from "../../layout/components/product-view";
interface Column {
  id: string;
  label: string;
  minWidth?: number;
  align?: "right";
  format?: (value: number) => string;
}

const columns: readonly Column[] = [
  { id: "id", label: "ID", minWidth: 80 },
  { id: "name", label: "Nombre", minWidth: 170 },
  { id: "info", label: "Información", minWidth: 100 },
  { id: "contact", label: "Contacto", minWidth: 100 },
  { id: "status", label: "Estado", minWidth: 100 },
];

const scrollGlobalStyles = (
  <GlobalStyles
    styles={{
      "*::-webkit-scrollbar-thumb": {
        borderRadius: "5px",
        backgroundColor: customPalette.paperDarkTransparent,
      },
      "*::-webkit-scrollbar": {
        width: 6,
      },
    }}
  />
);

export const ClientsTable = (props: any) => {
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);
  const [rows] = useState(props.data);
  const navigate = useNavigate();
  //   const [newProductModal, setNewProductModal] = React.useState(false);

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setRowsPerPage(+event.target.value);
    setPage(0);
  };
  const [open, setOpen] = useState(false);
  return (
    <>
      {scrollGlobalStyles}
      <Paper sx={{ width: "100%", overflow: "hidden" }}>
        <TableContainer
          sx={{
            maxHeight: 440,
          }}
        >
          <Box sx={{ width: "100%", p: 2 }}>
            <Typography variant="body1" gutterBottom>
              Clientes
            </Typography>
            <Divider />
            <Grid sx={{ mt: 1 }} container spacing={2}>
              <Grid item xs={12} md={6}>
                <Box>
                  <Button
                    sx={{
                      // marginLeft: 1,
                      marginY: { xs: 1, sm: 0 },
                    }}
                    variant="outlined"
                    startIcon={<AddIcon />}
                    onClick={() => navigate("new")}
                  >
                    Nuevo cliente
                  </Button>
                  <Box sx={{ display: "inline", ml: 1 }} />
                  <Button variant="contained">
                    <Edit />
                  </Button>
                  <Box sx={{ display: "inline", ml: 1 }} />
                  <Button variant="contained" onClick={() => setOpen(true)}>
                    <DeleteOutline />
                  </Button>
                </Box>
              </Grid>
              <Grid item xs={12} md={6}>
                <Box
                  sx={{
                    display: "flex",
                    alignItems: "flex-end",
                    position: "relative",
                  }}
                >
                  <TextField
                    fullWidth
                    size="small"
                    id="search-client"
                    label="Buscar cliente"
                    variant="outlined"
                  />
                  <SearchIcon
                    sx={{
                      position: "absolute",
                      right: 0,
                      top: 0,
                      transform: "translate(-50%, 20%)",
                      color: "action.active",
                      mr: 1,
                      my: 0.5,
                    }}
                  />
                </Box>
              </Grid>
            </Grid>
          </Box>
        </TableContainer>
        <TableContainer
          sx={{
            maxHeight: 440,
          }}
        >
          <Table stickyHeader aria-label="sticky table">
            <TableHead>
              <TableRow>
                {columns.map((column) => (
                  <TableCell
                    key={column.id}
                    align={column.align}
                    style={{ minWidth: column.minWidth }}
                    sx={{
                      ...(column.id !== "name" && {
                        // "&.MuiTableCell-root": {
                        //   backgroundColor: customPalette.paperDarkHalfSolid,
                        // },
                        display: { xs: "none", sm: "table-cell" },
                      }),
                    }}
                  >
                    {column.label}
                  </TableCell>
                ))}
                {/* <TableCell
                    sx={{ width: 40, display: { xs: "none", sm: "table-cell" } }}
                  />
                  <TableCell
                    sx={{ width: 40, display: { xs: "none", sm: "table-cell" } }}
                  />
                  <TableCell
                    sx={{ width: 40, display: { xs: "none", sm: "table-cell" } }}
                  /> */}
              </TableRow>
            </TableHead>
            <TableBody>
              {rows
                .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
                .map((row: any, i: number) => {
                  return (
                    <TableRow
                      hover
                      role="checkbox"
                      tabIndex={-1}
                      key={"row-" + i}
                    >
                      {columns.map((column) => {
                        const value = row[column.id];
                        return (
                          <TableCell
                            key={column.id}
                            align={column.align}
                            sx={{
                              ...(column.id !== "name" && {
                                display: { xs: "none", sm: "table-cell" },
                              }),
                            }}
                          >
                            {column.format && typeof value === "number"
                              ? column.format(value)
                              : value}
                          </TableCell>
                        );
                      })}
                    </TableRow>
                  );
                })}
            </TableBody>
          </Table>
        </TableContainer>
        <TablePagination
          rowsPerPageOptions={[10, 25, 100]}
          labelRowsPerPage="Resultados por página: "
          component="div"
          count={rows.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </Paper>
      <Outlet />
      <PopupMessage
        open={open}
        handleClose={() => {
          setOpen(false);
        }}
        message={
          <Typography color="white" variant="body1">
            Funcionalidad en construcción
          </Typography>
        }
      ></PopupMessage>
    </>
  );
};
