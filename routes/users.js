const express = require("express");
const router = express.Router();
const passport = require('passport');
const usersController = require('../controllers/users_controller');

router.get('/profile/:id',passport.checkAuthentication, usersController.profile);
router.get('/sign-in', usersController.signin);
router.get('/sign-up', usersController.signup);
router.post('/create', usersController.create);
router.post('/create-session', passport.authenticate('local', { failureRedirect:'/'  }), usersController.createSession);
router.get('/sign-out', usersController.destroySession);
router.post('/update/:id', passport.checkAuthentication, usersController.update);

module.exports = router;
