import {
  Card,
  Typography,
  CardContent,
  Box,
  Link,
  IconButton,
  CardMedia,
} from "@mui/material";

import {
  AttachMoney,
  EmojiPeople,
  DirectionsWalk,
  TrendingUp,
  WarningAmber,
} from "@mui/icons-material";
import SmallInfoCard from "./small-info-card";
import chart from "../../assets/Chart.png";

export const Dashboard = () => {
  return (
    <>
      <Card>
        <CardContent>
          <Typography gutterBottom variant="h5" component="h2">
            Bienvenido!
          </Typography>
          <br />
          <Typography variant="body2" color="textSecondary" component="p">
            La información más importante sobre tu producto aquí...
          </Typography>
          <br />
        </CardContent>

        <img src={chart} alt="chart" />
      </Card>

      <Box component="div" sx={{ display: "flex" }}>
        <Box
          component="div"
          sx={{
            display: "flex",
            flexWrap: "wrap",
          }}
        >
          <SmallInfoCard
            title="Productos nuevos"
            info={<InfoData data="+133" percentage={3} />}
            icon={
              <RoundIconCard>
                <TrendingUp fontSize="medium" sx={{ color: "#f0f0f0" }} />
              </RoundIconCard>
            }
          />
          <SmallInfoCard
            title="Nuevos clientes"
            info={<InfoData data="+41" percentage={-3} />}
            icon={
              <RoundIconCard>
                <EmojiPeople fontSize="medium" sx={{ color: "#f0f0f0" }} />
              </RoundIconCard>
            }
          />
          <SmallInfoCard
            title="Productos vendidos"
            info={<InfoData data="$13,419" percentage={5} />}
            icon={
              <RoundIconCard>
                <AttachMoney fontSize="medium" sx={{ color: "#f0f0f0" }} />
              </RoundIconCard>
            }
          />
          <SmallInfoCard
            title="Productos Agotados"
            info={<InfoData data="7" />}
            icon={
              <RoundIconCard>
                <WarningAmber fontSize="medium" sx={{ color: "#f0f0f0" }} />
              </RoundIconCard>
            }
          />
        </Box>
      </Box>
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

const TEST_GRADIENT =
  "radial-gradient(farthest-side at -10px -10px, #ddff33, #3E214C)";

const RoundIconCard = (props: any) => {
  return (
    <IconButton sx={{ background: TEST_GRADIENT }}>{props.children}</IconButton>
  );
};
