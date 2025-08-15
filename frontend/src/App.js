import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import 'bootstrap/dist/css/bootstrap.min.css';

import Navbar from "./components/Navbar";
import TruckList from "./components/TruckList";
import EditTruck from "./components/EditTruck";
import CreateTruck from "./components/CreateTruck";
import CreateExpense from "./components/CreateExpense";
import ExpenseList from "./components/ExpenseList";
import Dashboard from "./components/Dashboard";
import EditExpense from "./components/EditExpense";

function App() {
  return (
    <Router>
      <div className="container">
        <Navbar />
        <br/>
        <Routes>
          <Route path="/" element={<TruckList />} />
          <Route path="/edit/:id" element={<EditTruck />} />
          <Route path="/create" element={<CreateTruck />} />
          <Route path="/create-expense" element={<CreateExpense />} />
          <Route path="/truck/:truckId/expenses" element={<ExpenseList />} />
          <Route path="/edit-expense/:id" element={<EditExpense />} />
          <Route path="/dashboard" element={<Dashboard />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
