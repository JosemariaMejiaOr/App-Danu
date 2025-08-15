import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';

const EditTruck = () => {
    const [make, setMake] = useState('');
    const [model, setModel] = useState('');
    const [year, setYear] = useState('');
    const [vin, setVin] = useState('');
    const [licensePlate, setLicensePlate] = useState('');
    const [zone, setZone] = useState('');

    const { id } = useParams();

    useEffect(() => {
        axios.get('http://localhost:5000/trucks/'+id)
            .then(response => {
                setMake(response.data.make);
                setModel(response.data.model);
                setYear(response.data.year);
                setVin(response.data.vin);
                setLicensePlate(response.data.licensePlate);
                setZone(response.data.zone);
            })
            .catch(function (error) {
                console.log(error);
            })
    }, [id]);

    const onSubmit = (e) => {
        e.preventDefault();
        const truck = {
            make,
            model,
            year,
            vin,
            licensePlate,
            zone
        };
        console.log(truck);
        axios.post('http://localhost:5000/trucks/update/'+id, truck)
            .then(res => console.log(res.data));
        window.location = '/';
    }

    return (
        <div>
            <h3>Edit Truck</h3>
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
                    <input type="submit" value="Edit Truck" className="btn btn-primary" />
                </div>
            </form>
        </div>
    )
}

export default EditTruck;
