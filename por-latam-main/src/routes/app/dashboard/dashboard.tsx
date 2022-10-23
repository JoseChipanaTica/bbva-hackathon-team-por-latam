import { ArrowRightAlt, Info } from "@mui/icons-material";
import {
  Typography,
  Grid,
  Paper,
  Link,
  Box,
  Card,
  CardContent,
  Skeleton,
} from "@mui/material";

import Chart from "../chart";
import Deposits from "../deposits";
import Orders from "../orders";
import { CustomSelectInput } from "./custom-select-input";
import SmallInfoCard from "./small-info-card";

function Copyright(props: any) {
  return (
    <Typography
      variant="body2"
      color="text.secondary"
      align="center"
      {...props}
    >
      {"Copyright © "}
      <Link color="inherit" href="https://mui.com/">
        Your Website
      </Link>{" "}
      {new Date().getFullYear()}
      {"."}
    </Typography>
  );
}

export const Dashboard = () => {
  return (
    <>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6} lg={4}>
          <SmallInfoCard
            title="Info relevante 1"
            info={<InfoData data="2310" percentage={3} />}
            icon={<Info fontSize="large" color="primary" />}
          />
        </Grid>
        <Grid item xs={12} md={6} lg={4}>
          <SmallInfoCard
            title="Info relevante 2"
            info={<InfoData data="25" percentage={10} />}
            icon={<Info fontSize="large" color="primary" />}
          />
        </Grid>
        <Grid item xs={12} md={9} lg={4}>
          <SmallInfoCard
            title="Info relevante 3"
            info={<InfoData data="10" percentage={-20} />}
            icon={<Info fontSize="large" color="primary" />}
          />
        </Grid>
      </Grid>
      <Grid sx={{ mt: 1 }} container spacing={3}>
        {/* Charts */}
        <Grid item xs={12} md={6}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box
                    sx={{ display: "flex", justifyContent: "space-between" }}
                  >
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      gutterBottom
                    >
                      Comparación
                    </Typography>
                    <CustomSelectInput />
                  </Box>
                  <Skeleton variant="rectangular" height={200} />
                  <Typography variant="caption" color="text.secondary">
                    Gráfica 1
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Box
                    sx={{ display: "flex", justifyContent: "space-between" }}
                  >
                    <Typography variant="body2" gutterBottom>
                      Sexo
                    </Typography>
                    <Box
                      sx={{ display: "flex", justifyContent: "space-between" }}
                    >
                      <Typography variant="body2" gutterBottom>
                        Fecha 1
                      </Typography>
                      <ArrowRightAlt />
                      <Typography variant="body2" gutterBottom>
                        Fecha 2
                      </Typography>
                    </Box>
                  </Box>
                  <Box
                    sx={{ display: "flex", justifyContent: "space-between" }}
                  >
                    <Typography variant="body1">Mujer: 90</Typography>
                    <Typography variant="body1">Hombre: 65</Typography>
                    <Typography variant="body1">Otro: 78</Typography>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12}>
              <Card>
                <CardContent>
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    gutterBottom
                  >
                    Información complementaria
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Grid>
        <Grid item xs={12} md={6}>
          <Card>
            <CardContent>
              <Box sx={{ display: "flex", justifyContent: "space-between" }}>
                <Typography variant="body2" color="text.secondary" gutterBottom>
                  Resumen
                </Typography>
              </Box>
              <Skeleton variant="rectangular" height={500} />
              <Typography variant="caption" color="text.secondary">
                Gráfica 2
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </>
  );
};

// <Copyright sx={{ pt: 4 }} />

export const OldDashboard = () => {
  return (
    <>
      <Grid container spacing={3}>
        {/* Chart */}
        <Grid item xs={12} md={8} lg={9}>
          <Paper
            sx={{
              p: 2,
              display: "flex",
              flexDirection: "column",
              height: 240,
            }}
          >
            <Chart />
          </Paper>
        </Grid>
        {/* Recent Deposits */}
        <Grid item xs={12} md={4} lg={3}>
          <Paper
            sx={{
              p: 2,
              display: "flex",
              flexDirection: "column",
              height: 240,
            }}
          >
            <Deposits />
          </Paper>
        </Grid>
        {/* Recent Orders */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2, display: "flex", flexDirection: "column" }}>
            <Orders />
          </Paper>
        </Grid>
      </Grid>
      <Copyright sx={{ pt: 4 }} />
    </>
  );
};

interface TestInfoData {
  data: string;
  percentage?: number;
}

const InfoData = ({ data, percentage }: TestInfoData) => {
  const percentageInfo =
    typeof percentage === "number" ? (
      <Typography
        variant="body1"
        sx={{
          fontWeight: "bold",
          color: percentage > 0 ? "success.light" : "error.main",
          marginLeft: "3px",
        }}
      >
        {percentage > 0 ? "+" : ""}
        {percentage}%
      </Typography>
    ) : null;
  return (
    <Box sx={{ display: "flex", alignItems: "flex-end" }}>
      <Typography variant="h5" sx={{ color: "text.primary" }}>
        {data}
      </Typography>
      {percentageInfo}
    </Box>
  );
};
