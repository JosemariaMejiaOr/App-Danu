const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const expenseSchema = new Schema({
  truck: { type: mongoose.Schema.Types.ObjectId, ref: 'Truck', required: true },
  date: { type: Date, required: true },
  amount: { type: Number, required: true },
  category: { type: String, required: true },
  description: { type: String },
  imageUrl: { type: String },
}, {
  timestamps: true,
});

const Expense = mongoose.model('Expense', expenseSchema);

module.exports = Expense;
