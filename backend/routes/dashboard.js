const router = require('express').Router();
const Expense = require('../models/expense.model');
const Truck = require('../models/truck.model');

// Get total investment by truck
router.get('/stats/investment-by-truck', async (req, res) => {
    try {
        const investmentByTruck = await Expense.aggregate([
            {
                $group: {
                    _id: '$truck',
                    totalAmount: { $sum: '$amount' }
                }
            },
            {
                $lookup: {
                    from: 'trucks', // The collection name for the Truck model
                    localField: '_id',
                    foreignField: '_id',
                    as: 'truckDetails'
                }
            },
            {
                $unwind: '$truckDetails'
            },
            {
                $project: {
                    truck: '$truckDetails.make',
                    model: '$truckDetails.model',
                    licensePlate: '$truckDetails.licensePlate',
                    totalAmount: 1
                }
            }
        ]);
        res.json(investmentByTruck);
    } catch (err) {
        res.status(400).json('Error: ' + err);
    }
});

// Get total spending by zone
router.get('/stats/spending-by-zone', async (req, res) => {
    try {
        const spendingByZone = await Expense.aggregate([
            {
                $lookup: {
                    from: 'trucks',
                    localField: 'truck',
                    foreignField: '_id',
                    as: 'truckDetails'
                }
            },
            {
                $unwind: '$truckDetails'
            },
            {
                $group: {
                    _id: '$truckDetails.zone',
                    totalAmount: { $sum: '$amount' }
                }
            },
            {
                $project: {
                    zone: '$_id',
                    totalAmount: 1,
                    _id: 0
                }
            }
        ]);
        res.json(spendingByZone);
    } catch (err) {
        res.status(400).json('Error: ' + err);
    }
});

module.exports = router;
