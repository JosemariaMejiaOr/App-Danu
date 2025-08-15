import React from 'react';
import { Link } from 'react-router-dom';

const Navbar = () => {
  return (
    <nav className="navbar navbar-expand-lg navbar-dark bg-dark">
      <div className="container">
        <Link to="/" className="navbar-brand">Fleet Management</Link>
        <div className="collapse navbar-collapse">
          <ul className="navbar-nav mr-auto">
            <li className="nav-item">
              <Link to="/" className="nav-link">Trucks</Link>
            </li>
            <li className="nav-item">
              <Link to="/create" className="nav-link">Create Truck</Link>
            </li>
            <li className="nav-item">
              <Link to="/create-expense" className="nav-link">Create Expense</Link>
            </li>
            <li className="nav-item">
                <Link to="/dashboard" className="nav-link">Dashboard</Link>
            </li>
          </ul>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
