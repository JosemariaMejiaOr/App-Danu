import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { Bar, Pie } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement } from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend, ArcElement);

const Dashboard = () => {
    const [investmentByTruckData, setInvestmentByTruckData] = useState({ labels: [], datasets: [] });
    const [spendingByZoneData, setSpendingByZoneData] = useState({ labels: [], datasets: [] });

    useEffect(() => {
        // Fetch investment by truck
        axios.get('http://localhost:5000/dashboard/stats/investment-by-truck')
            .then(response => {
                const data = response.data;
                const labels = data.map(item => `${item.truck} ${item.model} (${item.licensePlate})`);
                const amounts = data.map(item => item.totalAmount);
                setInvestmentByTruckData({
                    labels,
                    datasets: [{
                        label: 'Total Investment by Truck',
                        data: amounts,
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1,
                    }]
                });
            })
            .catch(error => console.log(error));

        // Fetch spending by zone
        axios.get('http://localhost:5000/dashboard/stats/spending-by-zone')
            .then(response => {
                const data = response.data;
                const labels = data.map(item => item.zone);
                const amounts = data.map(item => item.totalAmount);
                setSpendingByZoneData({
                    labels,
                    datasets: [{
                        label: 'Spending by Zone',
                        data: amounts,
                        backgroundColor: [
                            'rgba(255, 99, 132, 0.6)',
                            'rgba(54, 162, 235, 0.6)',
                            'rgba(255, 206, 86, 0.6)',
                            'rgba(75, 192, 192, 0.6)',
                            'rgba(153, 102, 255, 0.6)',
                            'rgba(255, 159, 64, 0.6)',
                        ],
                    }]
                });
            })
            .catch(error => console.log(error));
    }, []);

    return (
        <div>
            <h3>Dashboard</h3>
            <div style={{ maxWidth: '800px', margin: 'auto' }}>
                <h4>Investment by Truck</h4>
                <Bar data={investmentByTruckData} options={{ responsive: true, plugins: { legend: { position: 'top' }, title: { display: true, text: 'Total Investment per Truck' } } }} />
            </div>
            <hr />
            <div style={{ maxWidth: '500px', margin: 'auto' }}>
                <h4>Spending by Zone</h4>
                <Pie data={spendingByZoneData} options={{ responsive: true, plugins: { legend: { position: 'top' }, title: { display: true, text: 'Total Spending per Zone' } } }} />
            </div>
        </div>
    );
}

export default Dashboard;
