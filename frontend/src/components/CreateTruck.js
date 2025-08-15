import React, { useState } from 'react';
import axios from 'axios';

const CreateTruck = () => {
  const [make, setMake] = useState('');
  const [model, setModel] = useState('');
  const [year, setYear] = useState('');
  const [vin, setVin] = useState('');
  const [licensePlate, setLicensePlate] = useState('');
  const [zone, setZone] = useState('');

  const onSubmit = (e) => {
    e.preventDefault();

    const newTruck = {
      make,
      model,
      year,
      vin,
      licensePlate,
      zone
    };

    axios.post('http://localhost:5000/trucks/add', newTruck)
      .then(res => console.log(res.data));

    // clear form
    setMake('');
    setModel('');
    setYear('');
    setVin('');
    setLicensePlate('');
    setZone('');
  }

  return (
    <div>
      <h3>Create New Truck</h3>
      <form onSubmit={onSubmit}>
        <div className="form-group">
          <label>Make: </label>
          <input type="text"
            required
            className="form-control"
            value={make}
            onChange={(e) => setMake(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Model: </label>
          <input type="text"
            required
            className="form-control"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Year: </label>
          <input type="text"
            required
            className="form-control"
            value={year}
            onChange={(e) => setYear(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>VIN: </label>
          <input type="text"
            required
            className="form-control"
            value={vin}
            onChange={(e) => setVin(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>License Plate: </label>
          <input type="text"
            required
            className="form-control"
            value={licensePlate}
            onChange={(e) => setLicensePlate(e.target.value)}
          />
        </div>
        <div className="form-group">
          <label>Zone: </label>
          <input type="text"
            required
            className="form-control"
            value={zone}
            onChange={(e) => setZone(e.target.value)}
          />
        </div>
        <div className="form-group">
          <input type="submit" value="Create Truck" className="btn btn-primary" />
        </div>
      </form>
    </div>
  )
}

export default CreateTruck;
