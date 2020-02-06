const express = require("express");
const router = express.Router();
const passport = require("passport");

const jobController = require('../controllers/job_controller');

router.post('/create_job', passport.checkAuthentication, jobController.create_job);
router.get('/create', passport.checkAuthentication, jobController.create);
router.get('/delete/:id', passport.checkAuthentication, jobController.delete);
//router.get ('/get_job', passport.checkAuthentication, jobController.get_job);

module.exports = router;