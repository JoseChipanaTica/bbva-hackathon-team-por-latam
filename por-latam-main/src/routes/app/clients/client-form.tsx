import { Box, Grid, TextField } from "@mui/material";

export const ClientForm = () => {
  return (
    <Grid container spacing={1}>
      <Grid item xs={12}>
        <TextField fullWidth variant="outlined" label="Nombre y apellido" />
      </Grid>
      <Grid item xs={12}>
        <TextField fullWidth variant="outlined" label="Contacto" />
      </Grid>
      <Grid item xs={12}>
        <TextField fullWidth variant="outlined" label="Información" />
      </Grid>
      <Grid item xs={12}>
        <TextField fullWidth variant="outlined" label="Estado" />
      </Grid>
    </Grid>
  );
};
