const express = require("express");
const router = express.Router();
const passport = require("passport");

const companyController = require('../controllers/company_controller');
router.get('/home',companyController.home);

module.exports = router;