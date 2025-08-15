const router = require('express').Router();
let Truck = require('../models/truck.model');

// Get all trucks
router.route('/').get((req, res) => {
  Truck.find()
    .then(trucks => res.json(trucks))
    .catch(err => res.status(400).json('Error: ' + err));
});

// Add a new truck
router.route('/add').post((req, res) => {
  const { make, model, year, vin, licensePlate, zone } = req.body;

  const newTruck = new Truck({
    make,
    model,
    year,
    vin,
    licensePlate,
    zone,
  });

  newTruck.save()
    .then(() => res.json('Truck added!'))
    .catch(err => res.status(400).json('Error: ' + err));
});

// Get a truck by ID
router.route('/:id').get((req, res) => {
  Truck.findById(req.params.id)
    .then(truck => res.json(truck))
    .catch(err => res.status(400).json('Error: ' + err));
});

// Update a truck by ID
router.route('/update/:id').post((req, res) => {
  Truck.findById(req.params.id)
    .then(truck => {
      truck.make = req.body.make || truck.make;
      truck.model = req.body.model || truck.model;
      truck.year = req.body.year || truck.year;
      truck.vin = req.body.vin || truck.vin;
      truck.licensePlate = req.body.licensePlate || truck.licensePlate;
      truck.zone = req.body.zone || truck.zone;

      truck.save()
        .then(() => res.json('Truck updated!'))
        .catch(err => res.status(400).json('Error: ' + err));
    })
    .catch(err => res.status(400).json('Error: ' + err));
});

// Delete a truck by ID
router.route('/:id').delete((req, res) => {
  Truck.findByIdAndDelete(req.params.id)
    .then(() => res.json('Truck deleted.'))
    .catch(err => res.status(400).json('Error: ' + err));
});

module.exports = router;
