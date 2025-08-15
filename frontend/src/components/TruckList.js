import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';

const TruckList = () => {
  const [trucks, setTrucks] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:5000/trucks/')
      .then(response => {
        setTrucks(response.data);
      })
      .catch((error) => {
        console.log(error);
      })
  }, []);

  const deleteTruck = (id) => {
    axios.delete('http://localhost:5000/trucks/'+id)
      .then(res => console.log(res.data));

    setTrucks(trucks.filter(el => el._id !== id));
  }

  const truckList = () => {
    return trucks.map(currenttruck => {
      return <Truck truck={currenttruck} deleteTruck={deleteTruck} key={currenttruck._id}/>;
    })
  }

  return (
    <div>
      <h3>Trucks</h3>
      <table className="table">
        <thead className="thead-light">
          <tr>
            <th>Make</th>
            <th>Model</th>
            <th>Year</th>
            <th>VIN</th>
            <th>License Plate</th>
            <th>Zone</th>
            <th>Actions</th>
          </tr>
        </thead>
        <tbody>
          {truckList()}
        </tbody>
      </table>
    </div>
  );
}

const Truck = props => (
  <tr>
    <td>{props.truck.make}</td>
    <td>{props.truck.model}</td>
    <td>{props.truck.year}</td>
    <td>{props.truck.vin}</td>
    <td>{props.truck.licensePlate}</td>
    <td>{props.truck.zone}</td>
    <td>
      <Link to={"/edit/"+props.truck._id}>edit</Link> | <a href="#" onClick={() => { props.deleteTruck(props.truck._id) }}>delete</a> | <Link to={"/truck/"+props.truck._id+"/expenses"}>expenses</Link>
    </td>
  </tr>
)

export default TruckList;
