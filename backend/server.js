require('dotenv').config();
const express = require('express');
const cors = require('cors');
const mongoose = require('mongoose');

const app = express();
const port = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

const uri = process.env.ATLAS_URI;
mongoose.connect(uri, { useNewUrlParser: true, useUnifiedTopology: true });
const connection = mongoose.connection;
connection.once('open', () => {
  console.log("MongoDB database connection established successfully");
})

const trucksRouter = require('./routes/trucks');
const expensesRouter = require('./routes/expenses');
const dashboardRouter = require('./routes/dashboard');

app.use('/trucks', trucksRouter);
app.use('/expenses', expensesRouter);
app.use('/dashboard', dashboardRouter);

app.get('/', (req, res) => {
  res.send('Hello from Fleet Management Backend!');
});

app.listen(port, () => {
  console.log(`Server is running on port: ${port}`);
});
