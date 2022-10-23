import { Card, CardContent, Box, Typography } from "@mui/material";

interface SmallInfoCardProps {
  title: string;
  info?: JSX.Element;
  icon?: JSX.Element;
}

export default function SmallInfoCard({
  title,
  info,
  icon,
}: SmallInfoCardProps) {
  return (
    <Card
    //   sx={{
    //     minWidth: 200,
    //     marginTop: 1,
    //     marginRight: 1,
    //     flexGrow: "1",
    //     flexBasis: "40%",
    //   }}
    >
      <CardContent sx={{ display: "flex", justifyContent: "space-between" }}>
        <Box component="div">
          <Typography
            variant="body2"
            sx={{ color: "text.disabled", fontWeight: "bold" }}
          >
            {title}
          </Typography>
          {info}
        </Box>
        <Box component="div">{icon}</Box>
      </CardContent>
    </Card>
  );
}
