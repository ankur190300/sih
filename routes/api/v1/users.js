const express = require("express");

const router = express.Router();
const User_api = require("../../../controllers/api/v1/users_api");

router.post('/create-session', User_api.createSession);
module.exports = router;