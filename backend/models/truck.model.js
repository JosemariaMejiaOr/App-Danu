const mongoose = require('mongoose');

const Schema = mongoose.Schema;

const truckSchema = new Schema({
  make: { type: String, required: true },
  model: { type: String, required: true },
  year: { type: Number, required: true },
  vin: { type: String, required: true, unique: true },
  licensePlate: { type: String, required: true, unique: true },
  zone: { type: String, required: true },
}, {
  timestamps: true,
});

const Truck = mongoose.model('Truck', truckSchema);

module.exports = Truck;
