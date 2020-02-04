const express = require("express");

const router = express.Router();

const homeController = require('../controllers/home_controller');

console.log('router is loaded');
router.get('/', homeController.home);
router.use('/api', require('./api'));
router.use('/users', require('./users'));
router.use('/posts', require('./posts'));
router.use('/comments', require('./comments.js'));
router.use('/likes', require('./likes.js'));
// added the job route
router.use('/job', require('./jobs.js'));
router.use('/company', require('./company'));

module.exports = router;
