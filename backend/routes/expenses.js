const router = require('express').Router();
let Expense = require('../models/expense.model');

// Get all expenses
router.route('/').get((req, res) => {
  Expense.find()
    .then(expenses => res.json(expenses))
    .catch(err => res.status(400).json('Error: ' + err));
});

// Add a new expense
router.route('/add').post((req, res) => {
  const { truck, date, amount, category, description, imageUrl } = req.body;

  const newExpense = new Expense({
    truck,
    date,
    amount,
    category,
    description,
    imageUrl,
  });

  newExpense.save()
    .then(() => res.json('Expense added!'))
    .catch(err => res.status(400).json('Error: ' + err));
});

// Get an expense by ID
router.route('/:id').get((req, res) => {
  Expense.findById(req.params.id)
    .then(expense => res.json(expense))
    .catch(err => res.status(400).json('Error: ' + err));
});

// Update an expense by ID
router.route('/update/:id').post((req, res) => {
  Expense.findById(req.params.id)
    .then(expense => {
      expense.truck = req.body.truck || expense.truck;
      expense.date = req.body.date || expense.date;
      expense.amount = req.body.amount || expense.amount;
      expense.category = req.body.category || expense.category;
      expense.description = req.body.description || expense.description;
      expense.imageUrl = req.body.imageUrl || expense.imageUrl;

      expense.save()
        .then(() => res.json('Expense updated!'))
        .catch(err => res.status(400).json('Error: ' + err));
    })
    .catch(err => res.status(400).json('Error: ' + err));
});

// Delete an expense by ID
router.route('/:id').delete((req, res) => {
  Expense.findByIdAndDelete(req.params.id)
    .then(() => res.json('Expense deleted.'))
    .catch(err => res.status(400).json('Error: ' + err));
});

// Get all expenses for a specific truck
router.route('/truck/:truckId').get((req, res) => {
  Expense.find({ truck: req.params.truckId })
    .then(expenses => res.json(expenses))
    .catch(err => res.status(400).json('Error: ' + err));
});

module.exports = router;
