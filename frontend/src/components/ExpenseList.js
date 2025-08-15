import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Link, useParams } from 'react-router-dom';

const ExpenseList = () => {
    const [expenses, setExpenses] = useState([]);
    const { truckId } = useParams();

    useEffect(() => {
        axios.get(`http://localhost:5000/expenses/truck/${truckId}`)
            .then(response => {
                setExpenses(response.data);
            })
            .catch((error) => {
                console.log(error);
            })
    }, [truckId]);

    const deleteExpense = (id) => {
        axios.delete('http://localhost:5000/expenses/'+id)
            .then(res => console.log(res.data));

        setExpenses(expenses.filter(el => el._id !== id));
    }

    const expenseList = () => {
        return expenses.map(currentexpense => {
            return <Expense expense={currentexpense} deleteExpense={deleteExpense} key={currentexpense._id}/>;
        })
    }

    return (
        <div>
            <h3>Expenses</h3>
            <table className="table">
                <thead className="thead-light">
                    <tr>
                        <th>Date</th>
                        <th>Amount</th>
                        <th>Category</th>
                        <th>Description</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {expenseList()}
                </tbody>
            </table>
        </div>
    );
}

const Expense = props => (
    <tr>
        <td>{new Date(props.expense.date).toLocaleDateString()}</td>
        <td>{props.expense.amount}</td>
        <td>{props.expense.category}</td>
        <td>{props.expense.description}</td>
        <td>
            <Link to={"/edit-expense/"+props.expense._id}>edit</Link> | <a href="#" onClick={() => { props.deleteExpense(props.expense._id) }}>delete</a>
        </td>
    </tr>
)

export default ExpenseList;
