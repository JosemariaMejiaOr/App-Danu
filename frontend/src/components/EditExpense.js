import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useParams, useNavigate } from 'react-router-dom';

const EditExpense = () => {
    const [truck, setTruck] = useState('');
    const [date, setDate] = useState('');
    const [amount, setAmount] = useState(0);
    const [category, setCategory] = useState('');
    const [description, setDescription] = useState('');

    const { id } = useParams();
    const navigate = useNavigate();

    useEffect(() => {
        axios.get('http://localhost:5000/expenses/'+id)
            .then(response => {
                setTruck(response.data.truck);
                setDate(new Date(response.data.date).toISOString().slice(0, 10));
                setAmount(response.data.amount);
                setCategory(response.data.category);
                setDescription(response.data.description);
            })
            .catch(function (error) {
                console.log(error);
            })
    }, [id]);

    const onSubmit = (e) => {
        e.preventDefault();
        const expense = {
            truck,
            date,
            amount,
            category,
            description,
        };
        axios.post('http://localhost:5000/expenses/update/'+id, expense)
            .then(res => {
                console.log(res.data)
                // Go back to the expense list for the truck
                navigate(`/truck/${truck}/expenses`);
            });
    }

    return (
        <div>
            <h3>Edit Expense</h3>
            <form onSubmit={onSubmit}>
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
                    <input type="submit" value="Edit Expense" className="btn btn-primary" />
                </div>
            </form>
        </div>
    )
}

export default EditExpense;
