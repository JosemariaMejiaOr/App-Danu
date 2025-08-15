import React, { useState, useEffect } from 'react';
import axios from 'axios';

const CreateExpense = (props) => {
    const [trucks, setTrucks] = useState([]);
    const [truck, setTruck] = useState('');
    const [date, setDate] = useState(new Date().toISOString().slice(0, 10));
    const [amount, setAmount] = useState(0);
    const [category, setCategory] = useState('');
    const [description, setDescription] = useState('');

    useEffect(() => {
        axios.get('http://localhost:5000/trucks/')
            .then(response => {
                if (response.data.length > 0) {
                    setTrucks(response.data.map(truck => truck));
                    setTruck(response.data[0]._id);
                }
            })
    }, []);

    const onSubmit = (e) => {
        e.preventDefault();
        const newExpense = {
            truck,
            date,
            amount,
            category,
            description,
        };

        axios.post('http://localhost:5000/expenses/add', newExpense)
            .then(res => console.log(res.data));

        window.location = '/';
    }

    return (
        <div>
            <h3>Create New Expense</h3>
            <form onSubmit={onSubmit}>
                <div className="form-group">
                    <label>Truck: </label>
                    <select
                        required
                        className="form-control"
                        value={truck}
                        onChange={(e) => setTruck(e.target.value)}>
                        {
                            trucks.map(function (truck) {
                                return <option
                                    key={truck._id}
                                    value={truck._id}>{truck.make} - {truck.model}
                                </option>;
                            })
                        }
                    </select>
                </div>
                <div className="form-group">
                    <label>Date: </label>
                    <div>
                        <input type="date"
                            required
                            className="form-control"
                            value={date}
                            onChange={(e) => setDate(e.target.value)}
                        />
                    </div>
                </div>
                <div className="form-group">
                    <label>Amount: </label>
                    <input type="text"
                        required
                        className="form-control"
                        value={amount}
                        onChange={(e) => setAmount(e.target.value)}
                    />
                </div>
                <div className="form-group">
                    <label>Category: </label>
                    <input type="text"
                        required
                        className="form-control"
                        value={category}
                        onChange={(e) => setCategory(e.target.value)}
                    />
                </div>
                <div className="form-group">
                    <label>Description: </label>
                    <input type="text"
                        className="form-control"
                        value={description}
                        onChange={(e) => setDescription(e.target.value)}
                    />
                </div>
                <div className="form-group">
                    <input type="submit" value="Create Expense" className="btn btn-primary" />
                </div>
            </form>
        </div>
    )
}

export default CreateExpense;
